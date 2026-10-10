use std::{fs, path::{Path,PathBuf}};
pub(crate) struct CheckpointLocation {
    path: PathBuf,
    #[cfg(unix)]
    directory: fs::File,
}

impl CheckpointLocation {
    pub(crate) fn checked(path: &Path) -> anyhow::Result<Self> {
        let parent = path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("checkpoint needs a parent directory"))?;
        #[cfg(unix)]
        let directory = {
            use std::os::fd::{AsRawFd, FromRawFd};
            anyhow::ensure!(
                parent.is_absolute(),
                "checkpoint directory must be absolute"
            );
            let mut directory = fs::File::open("/")?;
            for component in parent.components() {
                let std::path::Component::Normal(name) = component else {
                    continue;
                };
                use std::os::unix::ffi::OsStrExt;
                let name = std::ffi::CString::new(name.as_bytes())?;
                let fd = unsafe {
                    libc::openat(
                        directory.as_raw_fd(),
                        name.as_ptr(),
                        libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                    )
                };
                if fd < 0 {
                    return Err(std::io::Error::last_os_error().into());
                }
                directory = unsafe { fs::File::from_raw_fd(fd) };
            }
            directory
        };
        Ok(Self {
            path: path.to_owned(),
            #[cfg(unix)]
            directory,
        })
    }

    fn path(&self, partial: bool) -> PathBuf {
        if partial {
            self.path.with_extension("part")
        } else {
            self.path.clone()
        }
    }

    #[cfg(unix)]
    fn name(&self, partial: bool) -> anyhow::Result<std::ffi::CString> {
        use std::os::unix::ffi::OsStrExt;
        Ok(std::ffi::CString::new(
            self.path(partial)
                .file_name()
                .ok_or_else(|| anyhow::anyhow!("checkpoint needs a filename"))?
                .as_bytes(),
        )?)
    }

    pub(crate) fn open(&self, partial: bool, write: bool) -> anyhow::Result<fs::File> {
        #[cfg(unix)]
        let file = {
            use std::os::fd::{AsRawFd, FromRawFd};
            let name = self.name(partial)?;
            let flags = if write {
                libc::O_RDWR | libc::O_CREAT
            } else {
                libc::O_RDONLY
            };
            let fd = unsafe {
                libc::openat(
                    self.directory.as_raw_fd(),
                    name.as_ptr(),
                    flags | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC,
                    0o600,
                )
            };
            if fd < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            unsafe { fs::File::from_raw_fd(fd) }
        };
        #[cfg(not(unix))]
        let file = open_regular(&self.path(partial), write)?;
        anyhow::ensure!(
            file.metadata()?.is_file(),
            "checkpoint is not a regular file"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            anyhow::ensure!(
                file.metadata()?.nlink() == 1,
                "checkpoint must not be hard-linked"
            );
        }
        Ok(file)
    }

}
#[cfg(not(unix))]
fn open_regular(path: &Path, write: bool) -> anyhow::Result<fs::File> {
    if let Ok(meta) = fs::symlink_metadata(path) {
        anyhow::ensure!(
            meta.is_file() && !meta.file_type().is_symlink(),
            "checkpoint is not a regular file"
        );
    }
    let mut options = fs::OpenOptions::new();
    options.read(true).write(write).create(write);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x00200000); // FILE_FLAG_OPEN_REPARSE_POINT
    }
    let file = options.open(path)?;
    anyhow::ensure!(
        file.metadata()?.is_file(),
        "checkpoint is not a regular file"
    );
    Ok(file)
}fn safe_component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
}
pub(crate) fn validate_relative_file(path: &str) -> anyhow::Result<()> {
    anyhow::ensure!(path.split('/').all(safe_component), "expected a relative checkpoint filename");
    Ok(())
}