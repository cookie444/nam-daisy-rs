//! A tiny allocation-free JSON tokenizer, sufficient for NAM metadata.
//!
//! It is *not* a general-purpose parser: it exposes primitive readers plus a
//! recursive [`Json::skip_value`], and the schema is driven by the caller.

/// Errors from the minimal JSON reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonError {
    UnexpectedEof,
    UnexpectedByte,
    Expected(&'static str),
    InvalidNumber,
}

pub type Result<T> = core::result::Result<T, JsonError>;

/// Strip the null terminator and any trailing padding from metadata bytes.
pub fn metadata_slice(meta: &[u8]) -> &[u8] {
    match meta.iter().position(|&b| b == 0) {
        Some(nul) => meta.split_at(nul).0,
        None => meta,
    }
}

/// A cursor over UTF-8 JSON bytes.
pub struct Json<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Json<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Self { b, pos: 0 }
    }

    #[inline]
    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }

    #[inline]
    fn bump(&mut self) -> Option<u8> {
        let v = self.peek();
        if v.is_some() {
            self.pos += 1;
        }
        v
    }

    /// Consume a single structural byte (e.g. a comma) located by `peek_non_ws`.
    #[inline]
    pub fn bump_comma(&mut self) {
        self.pos += 1;
    }

    pub fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    pub fn expect(&mut self, byte: u8) -> Result<()> {
        self.skip_ws();
        match self.bump() {
            Some(c) if c == byte => Ok(()),
            Some(_) => Err(JsonError::UnexpectedByte),
            None => Err(JsonError::UnexpectedEof),
        }
    }

    pub fn peek_non_ws(&mut self) -> Result<u8> {
        self.skip_ws();
        self.peek().ok_or(JsonError::UnexpectedEof)
    }

    /// Parse a JSON string literal, returning the raw (unescaped) bytes.
    /// Escape sequences are handled for the common `\"`, `\\`, `\/`, `\n`, `\t`.
    pub fn parse_string(&mut self) -> Result<&'a [u8]> {
        self.skip_ws();
        if self.bump() != Some(b'"') {
            return Err(JsonError::Expected("string"));
        }
        let start = self.pos;
        let mut escaped = false;
        while let Some(c) = self.peek() {
            if escaped {
                escaped = false;
                self.pos += 1;
                continue;
            }
            match c {
                b'\\' => {
                    escaped = true;
                    self.pos += 1;
                }
                b'"' => {
                    let s = &self.b[start..self.pos];
                    self.pos += 1;
                    return Ok(s);
                }
                _ => self.pos += 1,
            }
        }
        Err(JsonError::UnexpectedEof)
    }

    fn number_token(&mut self) -> Result<&'a [u8]> {
        self.skip_ws();
        let start = self.pos;
        if matches!(self.peek(), Some(b'-') | Some(b'+')) {
            self.pos += 1;
        }
        while let Some(c) = self.peek() {
            let ok =
                c.is_ascii_digit() || c == b'.' || c == b'e' || c == b'E' || c == b'-' || c == b'+';
            if ok {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(JsonError::InvalidNumber);
        }
        Ok(&self.b[start..self.pos])
    }

    pub fn parse_f32(&mut self) -> Result<f32> {
        let tok = self.number_token()?;
        let s = core::str::from_utf8(tok).map_err(|_| JsonError::InvalidNumber)?;
        s.parse::<f32>().map_err(|_| JsonError::InvalidNumber)
    }

    pub fn parse_u16(&mut self) -> Result<u16> {
        let tok = self.number_token()?;
        let s = core::str::from_utf8(tok).map_err(|_| JsonError::InvalidNumber)?;
        s.parse::<u16>().map_err(|_| JsonError::InvalidNumber)
    }

    pub fn parse_bool(&mut self) -> Result<bool> {
        self.skip_ws();
        if self.b.starts_with_at(b"true", self.pos) {
            self.pos += 4;
            Ok(true)
        } else if self.b.starts_with_at(b"false", self.pos) {
            self.pos += 5;
            Ok(false)
        } else {
            Err(JsonError::Expected("bool"))
        }
    }

    pub fn parse_null(&mut self) -> Result<()> {
        if self.b.starts_with_at(b"null", self.pos) {
            self.pos += 4;
            Ok(())
        } else {
            Err(JsonError::Expected("null"))
        }
    }

    /// Recursively skip any JSON value.
    pub fn skip_value(&mut self) -> Result<()> {
        match self.peek_non_ws()? {
            b'"' => {
                self.parse_string()?;
            }
            b'{' => {
                self.expect(b'{')?;
                if self.peek_non_ws()? == b'}' {
                    self.bump();
                    return Ok(());
                }
                loop {
                    self.parse_string()?;
                    self.expect(b':')?;
                    self.skip_value()?;
                    match self.peek_non_ws()? {
                        b',' => {
                            self.bump();
                        }
                        b'}' => {
                            self.bump();
                            break;
                        }
                        _ => return Err(JsonError::Expected("',' or '}'")),
                    }
                }
            }
            b'[' => {
                self.expect(b'[')?;
                if self.peek_non_ws()? == b']' {
                    self.bump();
                    return Ok(());
                }
                loop {
                    self.skip_value()?;
                    match self.peek_non_ws()? {
                        b',' => {
                            self.bump();
                        }
                        b']' => {
                            self.bump();
                            break;
                        }
                        _ => return Err(JsonError::Expected("',' or ']'")),
                    }
                }
            }
            b't' => {
                self.parse_bool()?;
            }
            b'f' => {
                self.parse_bool()?;
            }
            b'n' => {
                self.parse_null()?;
            }
            _ => {
                self.number_token()?;
            }
        }
        Ok(())
    }
}

/// Small helper so we don't need `slice::starts_with` on a cursor offset.
trait StartsWithAt {
    fn starts_with_at(&self, needle: &[u8], at: usize) -> bool;
}

impl StartsWithAt for [u8] {
    fn starts_with_at(&self, needle: &[u8], at: usize) -> bool {
        match self.get(at..at + needle.len()) {
            Some(s) => s == needle,
            None => false,
        }
    }
}
