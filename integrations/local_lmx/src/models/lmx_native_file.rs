//! Native LMX v2 container I/O. Opening reads only the header and bounded JSON index;
//! payloads are read on demand, without a loader, extraction, or cache.

use super::lmx_native::{
    ALIGNMENT, Block, Buffer, HEADER_BYTES, METADATA_LIMIT, Manifest, VERSION,
};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{self, Seek, SeekFrom, Write},
    path::Path,
};

const MAGIC: &[u8; 8] = b"LOOMLM2\0";
const CHUNK_BYTES: usize = 64 << 10;

#[derive(Debug)]
pub struct NativeFile {
    pub manifest: Manifest,
    pub identity: String,
    pub bytes: u64,
    file: File,
}

impl NativeFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("open native LMX {}", path.display()))?;
        let metadata = file.metadata()?;
        ensure!(metadata.is_file(), "native LMX must be a regular file");
        let bytes = metadata.len();
        ensure!(bytes >= HEADER_BYTES, "truncated native LMX header");
        let mut header = [0u8; HEADER_BYTES as usize];
        read_exact_at(&file, &mut header, 0).context("read native LMX header")?;
        ensure!(&header[..8] == MAGIC, "invalid native LMX magic");
        ensure!(
            u32::from_le_bytes(header[8..12].try_into()?) == VERSION,
            "unsupported native LMX version"
        );
        ensure!(
            u32::from_le_bytes(header[12..16].try_into()?) as u64 == HEADER_BYTES,
            "invalid native LMX header length"
        );
        ensure!(
            header[72..].iter().all(|byte| *byte == 0),
            "nonzero native LMX reserved header bytes"
        );
        let index_offset = u64::from_le_bytes(header[16..24].try_into()?);
        let index_bytes = u64::from_le_bytes(header[24..32].try_into()?);
        let declared_bytes = u64::from_le_bytes(header[32..40].try_into()?);
        ensure!(declared_bytes == bytes, "native LMX file length mismatch");
        ensure!(
            index_offset >= HEADER_BYTES
                && index_offset % ALIGNMENT == 0
                && index_bytes > 0
                && index_bytes <= METADATA_LIMIT
                && index_offset.checked_add(index_bytes) == Some(bytes),
            "invalid native LMX index range or EOF"
        );
        let mut index = vec![0u8; usize::try_from(index_bytes)?];
        read_exact_at(&file, &mut index, index_offset).context("read native LMX index")?;
        ensure!(
            Sha256::digest(&index)[..] == header[40..72],
            "native LMX index checksum mismatch"
        );
        let manifest: Manifest =
            serde_json::from_slice(&index).context("decode native LMX index")?;
        ensure!(
            manifest.version == VERSION,
            "unsupported native LMX manifest version"
        );
        manifest.validate(index_offset)?;
        let identity = manifest.base_identity()?;
        Ok(Self {
            manifest,
            identity,
            bytes,
            file,
        })
    }

    /// Reads exactly one declared block, without moving the file's shared cursor.
    /// On checksum failure the caller must discard the contents of `out`.
    pub fn read_block_into(&self, id: &str, out: &mut [u8]) -> Result<()> {
        let block = self
            .manifest
            .blocks
            .get(id)
            .with_context(|| format!("unknown native LMX block {id}"))?;
        ensure!(
            u64::try_from(out.len())? == block.bytes,
            "native LMX block {id} output size mismatch"
        );
        let mut hasher = Sha256::new();
        let mut offset = block.offset;
        for chunk in out.chunks_mut(CHUNK_BYTES) {
            read_exact_at(&self.file, chunk, offset)
                .with_context(|| format!("read native LMX block {id}"))?;
            hasher.update(&*chunk);
            offset = offset
                .checked_add(chunk.len() as u64)
                .context("native LMX block offset overflow")?;
        }
        check_hash(hasher, block).with_context(|| format!("native LMX block {id}"))
    }

    /// Reads a block-relative range directly into caller-owned storage. This does
    /// not authenticate a partial block; callers can verify the whole block first.
    pub fn read_range_into(&self, range: &Buffer, out: &mut [u8]) -> Result<()> {
        ensure!(
            u64::try_from(out.len())? == range.bytes,
            "native LMX range output size mismatch"
        );
        let block = self
            .manifest
            .blocks
            .get(&range.block)
            .with_context(|| format!("unknown native LMX block {}", range.block))?;
        ensure!(
            range
                .offset
                .checked_add(range.bytes)
                .is_some_and(|end| end <= block.bytes),
            "native LMX range escapes block {}",
            range.block
        );
        let mut offset = block
            .offset
            .checked_add(range.offset)
            .context("native LMX range offset overflow")?;
        for chunk in out.chunks_mut(CHUNK_BYTES) {
            read_exact_at(&self.file, chunk, offset)
                .with_context(|| format!("read native LMX range in {}", range.block))?;
            offset = offset
                .checked_add(chunk.len() as u64)
                .context("native LMX range offset overflow")?;
        }
        Ok(())
    }

    /// Streams all blocks with a fixed 64 KiB buffer. Progress counts payload bytes.
    pub fn verify(&self, progress: &mut dyn FnMut(u64, u64) -> bool) -> Result<()> {
        let total = payload_bytes(&self.manifest)?;
        let mut done = 0;
        check_progress(progress, done, total)?;
        let mut buffer = [0u8; CHUNK_BYTES];
        for (id, block) in &self.manifest.blocks {
            let mut hasher = Sha256::new();
            let mut consumed = 0;
            while consumed < block.bytes {
                let length = (block.bytes - consumed).min(CHUNK_BYTES as u64) as usize;
                let offset = block
                    .offset
                    .checked_add(consumed)
                    .context("native LMX block offset overflow")?;
                read_exact_at(&self.file, &mut buffer[..length], offset)
                    .with_context(|| format!("verify native LMX block {id}"))?;
                hasher.update(&buffer[..length]);
                consumed += length as u64;
                done += length as u64;
                check_progress(progress, done, total)?;
            }
            check_hash(hasher, block).with_context(|| format!("native LMX block {id}"))?;
        }
        check_progress(progress, done, total)
    }
}

/// Writes a new container in the destination directory and publishes without replacement.
/// The emitter receives block IDs in sorted order and must emit exactly `Block::bytes`.
/// Cancellation, including the final progress callback, leaves no destination file.
pub fn write(
    path: &Path,
    mut manifest: Manifest,
    emit: &mut dyn FnMut(&str, &mut dyn Write) -> Result<()>,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<NativeFile> {
    ensure!(
        manifest.version == VERSION,
        "unsupported native LMX manifest version"
    );
    match fs::symlink_metadata(path) {
        Ok(_) => anyhow::bail!("native LMX destination already exists: {}", path.display()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("check native LMX destination"),
    }
    let total = payload_bytes(&manifest)?;
    let mut done = 0;
    check_progress(progress, done, total)?;
    // Compute and validate the complete layout before calling the source emitter.
    let mut cursor = HEADER_BYTES;
    for block in manifest.blocks.values_mut() {
        block.offset = aligned(cursor)?;
        block.sha256 = "0".repeat(64);
        cursor = block
            .offset
            .checked_add(block.bytes)
            .context("native LMX block size overflow")?;
    }
    let index_offset = aligned(cursor)?;
    manifest.validate(index_offset)?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temporary =
        tempfile::NamedTempFile::new_in(parent).context("create native LMX temporary file")?;
    temporary.write_all(&[0u8; HEADER_BYTES as usize])?;
    for (id, block) in &mut manifest.blocks {
        temporary.seek(SeekFrom::Start(block.offset))?;
        let mut writer = BlockWriter {
            file: temporary.as_file_mut(),
            expected: block.bytes,
            written: 0,
            hasher: Sha256::new(),
            done: &mut done,
            total,
            progress,
            failure: None,
        };
        emit(id, &mut writer).with_context(|| format!("emit native LMX block {id}"))?;
        block.sha256 = writer
            .finish()
            .with_context(|| format!("emit native LMX block {id}"))?;
    }
    manifest.validate(index_offset)?;
    let mut index = IndexWriter(Vec::new());
    serde_json::to_writer(&mut index, &manifest).context("encode native LMX index")?;
    let index = index.0;
    let index_bytes = u64::try_from(index.len())?;
    let bytes = index_offset
        .checked_add(index_bytes)
        .context("native LMX file size overflow")?;
    let mut header = [0u8; HEADER_BYTES as usize];
    header[..8].copy_from_slice(MAGIC);
    header[8..12].copy_from_slice(&VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&(HEADER_BYTES as u32).to_le_bytes());
    header[16..24].copy_from_slice(&index_offset.to_le_bytes());
    header[24..32].copy_from_slice(&index_bytes.to_le_bytes());
    header[32..40].copy_from_slice(&bytes.to_le_bytes());
    header[40..72].copy_from_slice(&Sha256::digest(&index));
    temporary.seek(SeekFrom::Start(index_offset))?;
    for chunk in index.chunks(CHUNK_BYTES) {
        temporary.write_all(chunk)?;
        check_progress(progress, done, total)?;
    }
    temporary.as_file().set_len(bytes)?;
    temporary.seek(SeekFrom::Start(0))?;
    temporary.write_all(&header)?;
    temporary
        .as_file()
        .sync_all()
        .context("sync native LMX temporary file")?;
    let native =
        NativeFile::open(temporary.path()).context("validate completed native LMX file")?;
    check_progress(progress, done, total)?;
    // Discard the PersistError's tempfile immediately on failure, so RAII cleanup
    // is complete even if the caller retains the returned error.
    temporary
        .persist_noclobber(path)
        .map_err(|error| error.error)
        .with_context(|| format!("publish native LMX without overwriting {}", path.display()))?;
    Ok(native)
}

fn aligned(value: u64) -> Result<u64> {
    Ok(value
        .checked_add(ALIGNMENT - 1)
        .context("native LMX alignment overflow")?
        / ALIGNMENT
        * ALIGNMENT)
}

fn payload_bytes(manifest: &Manifest) -> Result<u64> {
    manifest.blocks.values().try_fold(0u64, |sum, block| {
        sum.checked_add(block.bytes)
            .context("native LMX payload size overflow")
    })
}

fn check_progress(progress: &mut dyn FnMut(u64, u64) -> bool, done: u64, total: u64) -> Result<()> {
    ensure!(progress(done, total), "native LMX operation cancelled");
    Ok(())
}

fn check_hash(hasher: Sha256, block: &Block) -> Result<()> {
    ensure!(
        format!("{:x}", hasher.finalize()).eq_ignore_ascii_case(&block.sha256),
        "checksum mismatch"
    );
    Ok(())
}

// Positional reads keep independent consumers from racing a shared seek cursor.
fn read_exact_at(file: &File, mut out: &mut [u8], mut offset: u64) -> io::Result<()> {
    #[cfg(not(any(unix, windows)))]
    let mut reader = {
        let mut reader = file.try_clone()?;
        reader.seek(SeekFrom::Start(offset))?;
        reader
    };
    while !out.is_empty() {
        #[cfg(unix)]
        let result = std::os::unix::fs::FileExt::read_at(file, out, offset);
        #[cfg(windows)]
        let result = std::os::windows::fs::FileExt::seek_read(file, out, offset);
        #[cfg(not(any(unix, windows)))]
        let result = std::io::Read::read(&mut reader, out);
        match result {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "truncated native LMX data",
                ));
            }
            Ok(length) => {
                offset = offset.checked_add(length as u64).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "native LMX read offset overflow",
                    )
                })?;
                out = &mut out[length..];
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

struct IndexWriter(Vec<u8>);

impl Write for IndexWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() as u64 > METADATA_LIMIT.saturating_sub(self.0.len() as u64) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "native LMX index exceeds metadata limit",
            ));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct BlockWriter<'a> {
    file: &'a mut File,
    expected: u64,
    written: u64,
    hasher: Sha256,
    done: &'a mut u64,
    total: u64,
    progress: &'a mut dyn FnMut(u64, u64) -> bool,
    // An emitter cannot turn an overrun, cancellation, or I/O failure into success
    // by swallowing the error and returning Ok(()).
    failure: Option<(io::ErrorKind, String)>,
}

impl BlockWriter<'_> {
    fn fail(&mut self, error: io::Error) -> io::Error {
        let (kind, message) = self
            .failure
            .get_or_insert_with(|| (error.kind(), error.to_string()));
        io::Error::new(*kind, message.clone())
    }

    fn finish(self) -> Result<String> {
        if let Some((kind, message)) = self.failure {
            return Err(io::Error::new(kind, message).into());
        }
        ensure!(
            self.written == self.expected,
            "native LMX block size mismatch: expected {}, emitted {}",
            self.expected,
            self.written
        );
        Ok(format!("{:x}", self.hasher.finalize()))
    }
}

impl Write for BlockWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if let Some((kind, message)) = &self.failure {
            return Err(io::Error::new(*kind, message.clone()));
        }
        if bytes.len() as u64 > self.expected - self.written {
            return Err(self.fail(io::Error::new(
                io::ErrorKind::InvalidInput,
                "native LMX block overrun",
            )));
        }
        for chunk in bytes.chunks(CHUNK_BYTES) {
            if let Err(error) = self.file.write_all(chunk) {
                return Err(self.fail(error));
            }
            self.hasher.update(chunk);
            self.written += chunk.len() as u64;
            *self.done += chunk.len() as u64;
            if !(self.progress)(*self.done, self.total) {
                return Err(self.fail(io::Error::other("native LMX operation cancelled")));
            }
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some((kind, message)) = &self.failure {
            return Err(io::Error::new(*kind, message.clone()));
        }
        self.file.flush().map_err(|error| self.fail(error))
    }
}

