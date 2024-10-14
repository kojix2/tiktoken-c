use log::warn;
use std::ffi::CStr;
use std::os::raw::c_char;

pub fn get_string_from_c_char(ptr: *const c_char) -> Result<String, std::str::Utf8Error> {
    let c_str = unsafe { CStr::from_ptr(ptr) };
    let str_slice = c_str.to_str()?;
    Ok(str_slice.to_string())
}

pub fn c_str_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }

    let c_str = match get_string_from_c_char(ptr) {
        Ok(str) => str,
        Err(_) => {
            warn!("Invalid UTF-8 sequence provided!");
            return None;
        }
    };

    Some(c_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_get_string_from_c_char() {
        let c_str = CString::new("I am a cat.").unwrap();
        let str = get_string_from_c_char(c_str.as_ptr()).unwrap();
        assert_eq!(str, "I am a cat.");
    }

    #[test]
    fn test_c_str_to_string() {
        let c_str = CString::new("I am a cat.").unwrap();
        let str = c_str_to_string(c_str.as_ptr()).unwrap();
        assert_eq!(str, "I am a cat.");
    }
}
