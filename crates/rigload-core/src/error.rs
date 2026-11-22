use std::fmt;
use std::path::PathBuf;

/// Every error carries the file it is about, so a failing sample can be found.
#[derive(Debug)]
pub enum Error {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Manifest {
        path: PathBuf,
        line: usize,
        msg: String,
    },
    Decode {
        path: PathBuf,
        msg: String,
    },
    /// Invalid configuration or parameters (Python: ValueError).
    Config(String),
    /// A worker thread that panicked: no single file to blame.
    Loader(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Error::Manifest { path, line, msg } => {
                write!(f, "{} line {line}: {msg}", path.display())
            }
            Error::Decode { path, msg } => write!(f, "{}: {msg}", path.display()),
            Error::Config(msg) | Error::Loader(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
