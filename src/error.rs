#[derive(Debug)]
pub struct RbpmaError {
    message: String,
}

impl RbpmaError {
    pub fn new(error_msg: &str) -> Self {
        Self {
            message: error_msg.to_string(),
        }
    }
}

impl std::fmt::Display for RbpmaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RbpmaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }

    fn description(&self) -> &str {
        "description() is deprecated; use Display"
    }

    fn cause(&self) -> Option<&dyn std::error::Error> {
        self.source()
    }

    fn provide<'a>(&'a self, request: &mut std::error::Request<'a>) {
        request.provide_ref(&self.message);
    }
}
