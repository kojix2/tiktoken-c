#[cfg(feature = "logging")]
use log::warn;
use std::ffi::{c_char, CStr};
use tiktoken_rs::CoreBPE;

// get_bpe_from_tokenizer is not yet implemented.
// Use tiktoken_r50k_base(), tiktoken_p50k_base(), tiktoken_p50k_edit(), tiktoken_cl100k_base(), and tiktoken_o200k_base()
// instead.

#[no_mangle]
pub extern "C" fn tiktoken_r50k_base() -> *mut CoreBPE {
    let bpe = tiktoken_rs::r50k_base();
    let corebpe = bpe.unwrap();
    let boxed = Box::new(corebpe);
    Box::into_raw(boxed)
}

#[no_mangle]
pub extern "C" fn tiktoken_p50k_base() -> *mut CoreBPE {
    let bpe = tiktoken_rs::p50k_base();
    let corebpe = bpe.unwrap();
    let boxed = Box::new(corebpe);
    Box::into_raw(boxed)
}

#[no_mangle]
pub extern "C" fn tiktoken_p50k_edit() -> *mut CoreBPE {
    let bpe = tiktoken_rs::p50k_edit();
    let corebpe = bpe.unwrap();
    let boxed = Box::new(corebpe);
    Box::into_raw(boxed)
}

#[no_mangle]
pub extern "C" fn tiktoken_cl100k_base() -> *mut CoreBPE {
    let bpe = tiktoken_rs::cl100k_base();
    let corebpe = bpe.unwrap();
    let boxed = Box::new(corebpe);
    Box::into_raw(boxed)
}

#[no_mangle]
pub extern "C" fn tiktoken_o200k_base() -> *mut CoreBPE {
    let bpe = tiktoken_rs::o200k_base();
    let corebpe = bpe.unwrap();
    let boxed = Box::new(corebpe);
    Box::into_raw(boxed)
}

#[no_mangle]
pub extern "C" fn tiktoken_destroy_corebpe(ptr: *mut CoreBPE) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(ptr);
    }
}

#[no_mangle]
pub extern "C" fn tiktoken_get_bpe_from_model(model: *const c_char) -> *mut CoreBPE {
    if model.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for model!");
        return std::ptr::null_mut();
    }
    let model = unsafe {
        let raw = CStr::from_ptr(model);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                #[cfg(feature = "logging")]
                warn!("Invalid UTF-8 sequence provided for model!");
                return std::ptr::null_mut();
            }
        }
    };
    let bpe = tiktoken_rs::get_bpe_from_model(model);
    match bpe {
        Ok(bpe) => {
            let boxed = Box::new(bpe);
            Box::into_raw(boxed)
        }
        Err(_) => {
            #[cfg(feature = "logging")]
            warn!("Failed to get BPE from model!");
            std::ptr::null_mut()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_c50k_base() {
        let corebpe = tiktoken_r50k_base();
        assert!(!corebpe.is_null());
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_p50k_base() {
        let corebpe = tiktoken_p50k_base();
        assert!(!corebpe.is_null());
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_p50k_edit() {
        let corebpe = tiktoken_p50k_edit();
        assert!(!corebpe.is_null());
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_cl100k_base() {
        let corebpe = tiktoken_cl100k_base();
        assert!(!corebpe.is_null());
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_o200k_base() {
        let corebpe = tiktoken_o200k_base();
        assert!(!corebpe.is_null());
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_get_bpe_from_model() {
        let model = CString::new("gpt-4").unwrap();
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        assert!(!corebpe.is_null());
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_get_bpe_from_model_invalid_model() {
        let model = CString::new("cat-gpt").unwrap();
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        assert!(corebpe.is_null());
    }
}
