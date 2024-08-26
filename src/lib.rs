use log::warn;
use simple_logger::SimpleLogger;
use std::ffi::CStr;
use std::os::raw::c_char;
use tiktoken_rs;
use tiktoken_rs::CoreBPE;

#[no_mangle]
pub extern "C" fn tiktoken_init_logger() {
    SimpleLogger::new().init().unwrap();
}

fn get_string_from_c_char(ptr: *const c_char) -> Result<String, std::str::Utf8Error> {
    let c_str = unsafe { CStr::from_ptr(ptr) };
    let str_slice = c_str.to_str()?;
    Ok(str_slice.to_string())
}

fn c_str_to_string(ptr: *const c_char) -> Option<String> {
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

// get_bpe_from_tokenizer is not yet implemented.
// Use tiktoken_r50k_base(), tiktoken_p50k_base(), tiktoken_p50k_edit(), tiktoken_cl100k_base(), and tiktoken_o200k_base()
// instead.

#[no_mangle]
pub extern "C" fn tiktoken_get_bpe_from_model(model: *const c_char) -> *mut CoreBPE {
    if model.is_null() {
        warn!("Null pointer provided for model!");
        return std::ptr::null_mut();
    }
    let model = unsafe {
        let raw = CStr::from_ptr(model);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
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
            warn!("Failed to get BPE from model!");
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tiktoken_get_completion_max_tokens(
    model: *const c_char,
    prompt: *const c_char,
) -> usize {
    if model.is_null() {
        warn!("Null pointer provided for model!");
        return usize::MAX;
    }
    if prompt.is_null() {
        warn!("Null pointer provided for prompt!");
        return usize::MAX;
    }
    let model = unsafe {
        let raw = CStr::from_ptr(model);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                warn!("Invalid UTF-8 sequence provided for model!");
                return usize::MAX;
            }
        }
    };
    let prompt = unsafe {
        let raw = CStr::from_ptr(prompt);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                warn!("Invalid UTF-8 sequence provided for prompt!");
                return usize::MAX;
            }
        }
    };
    match tiktoken_rs::get_completion_max_tokens(model, prompt) {
        Ok(max_tokens) => max_tokens,
        Err(_) => {
            warn!("Failed to get completion max tokens!");
            return usize::MAX;
        }
    }
}

#[repr(C)]
pub struct CFunctionCall {
    pub name: *const c_char,
    pub arguments: *const c_char,
}

#[repr(C)]
pub struct CChatCompletionRequestMessage {
    pub role: *const c_char,
    pub content: *const c_char,
    pub name: *const c_char,
    pub function_call: *const CFunctionCall,
}

#[no_mangle]
pub extern "C" fn tiktoken_num_tokens_from_messages(
    model: *const c_char,
    num_messages: u32,
    messages: *const CChatCompletionRequestMessage,
) -> usize {
    if model.is_null() {
        warn!("Null pointer provided for model!");
        return usize::MAX;
    }
    if messages.is_null() {
        warn!("Null pointer provided for messages!");
        return usize::MAX;
    }
    let model = unsafe {
        let raw = CStr::from_ptr(model);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                warn!("Invalid UTF-8 sequence provided for model!");
                return usize::MAX;
            }
        }
    };
    let messages = unsafe {
        let slice = std::slice::from_raw_parts(messages, num_messages as usize);
        let mut messages_vec = Vec::with_capacity(num_messages as usize);
        for message in slice {
            let role = c_str_to_string(message.role.clone()).unwrap_or_default();
            let content = c_str_to_string(message.content.clone());
            let name = c_str_to_string(message.name.clone());
            let function_call = if !message.function_call.is_null() {
                let fun_call = message.function_call;
                let fun_name = c_str_to_string((*fun_call).name).unwrap_or_default();
                let fun_args = c_str_to_string((*fun_call).arguments).unwrap_or_default();
                Some(tiktoken_rs::FunctionCall {
                    name: fun_name,
                    arguments: fun_args,
                })
            } else {
                None
            };
            messages_vec.push(tiktoken_rs::ChatCompletionRequestMessage {
                role: role,
                content: content,
                name: name,
                function_call: function_call,
            });
        }
        messages_vec
    };
    match tiktoken_rs::num_tokens_from_messages(model, &messages) {
        Ok(num_tokens) => num_tokens,
        Err(_) => {
            warn!("Failed to get num tokens!");
            return usize::MAX;
        }
    }
}

#[no_mangle]
pub extern "C" fn tiktoken_get_chat_completion_max_tokens(
    model: *const c_char,
    num_messages: u32,
    messages: *const CChatCompletionRequestMessage,
) -> usize {
    if model.is_null() {
        warn!("Null pointer provided for model!");
        return usize::MAX;
    }
    if messages.is_null() {
        warn!("Null pointer provided for messages!");
        return usize::MAX;
    }
    let model = unsafe {
        let raw = CStr::from_ptr(model);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                warn!("Invalid UTF-8 sequence provided for model!");
                return usize::MAX;
            }
        }
    };
    let messages = unsafe {
        let slice = std::slice::from_raw_parts(messages, num_messages as usize);
        let mut messages_vec = Vec::with_capacity(num_messages as usize);
        for message in slice {
            if message.role.is_null() {
                warn!("Null pointer provided for role!");
                return usize::MAX;
            }
            let role = c_str_to_string(message.role.clone()).unwrap_or_default();
            let content = c_str_to_string(message.content.clone());
            let name = c_str_to_string(message.name.clone());
            let function_call = if !message.function_call.is_null() {
                let fun_call = message.function_call;
                let fun_name = c_str_to_string((*fun_call).name).unwrap_or_default();
                let fun_args = c_str_to_string((*fun_call).arguments).unwrap_or_default();
                Some(tiktoken_rs::FunctionCall {
                    name: fun_name,
                    arguments: fun_args,
                })
            } else {
                None
            };
            messages_vec.push(tiktoken_rs::ChatCompletionRequestMessage {
                role: role,
                content: content,
                name: name,
                function_call: function_call,
            });
        }
        messages_vec
    };
    match tiktoken_rs::get_chat_completion_max_tokens(model, &messages) {
        Ok(max_tokens) => max_tokens,
        Err(_) => {
            warn!("Failed to get max tokens!");
            return usize::MAX;
        }
    }
}

#[no_mangle]
pub extern "C" fn tiktoken_corebpe_encode_ordinary(
    ptr: *mut CoreBPE,
    text: *const c_char,
    num_tokens: *mut usize,
) -> *mut usize {
    if ptr.is_null() {
        warn!("Null pointer provided for CoreBPE!");
        return std::ptr::null_mut();
    }
    if text.is_null() {
        warn!("Null pointer provided for text!");
        return std::ptr::null_mut();
    }
    let text = unsafe {
        let raw = CStr::from_ptr(text);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                warn!("Invalid UTF-8 sequence provided for text!");
                return std::ptr::null_mut();
            }
        }
    };
    let corebpe = unsafe { &mut *ptr };
    let encoded = corebpe.encode_ordinary(text);
    unsafe {
        if !num_tokens.is_null() {
            *num_tokens = encoded.len();
        }
    };
    let boxed = encoded.into_boxed_slice();
    Box::into_raw(boxed) as *mut usize
}

// pub fn encode(&self, text: &str, allowed_special: HashSet<&str>) -> Vec<usize>
#[no_mangle]
pub extern "C" fn tiktoken_corebpe_encode(
    ptr: *mut CoreBPE,
    text: *const c_char,
    allowed_special: *const *const c_char,
    allowed_special_len: usize,
    num_tokens: *mut usize,
) -> *mut usize {
    if ptr.is_null() {
        warn!("Null pointer provided for CoreBPE!");
        return std::ptr::null_mut();
    }
    if text.is_null() {
        warn!("Null pointer provided for text!");
        return std::ptr::null_mut();
    }
    if allowed_special.is_null() {
        warn!("Null pointer provided for allowed_special!");
        return std::ptr::null_mut();
    }
    let text = unsafe {
        let raw = CStr::from_ptr(text);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                warn!("Invalid UTF-8 sequence provided for text!");
                return std::ptr::null_mut();
            }
        }
    };
    let allowed_special = unsafe {
        let slice = std::slice::from_raw_parts(allowed_special, allowed_special_len);
        let mut allowed_special_hash_set = std::collections::HashSet::new();
        for i in 0..allowed_special_len {
            let c_str = CStr::from_ptr(slice[i]);
            let str_slice = match c_str.to_str() {
                Ok(valid_str) => valid_str,
                Err(_) => {
                    warn!("Invalid UTF-8 sequence provided for allowed_special!");
                    return std::ptr::null_mut();
                }
            };
            allowed_special_hash_set.insert(str_slice);
        }
        allowed_special_hash_set
    };
    let corebpe = unsafe { &mut *ptr };
    let encoded = corebpe.encode(text, allowed_special);
    unsafe {
        if !num_tokens.is_null() {
            *num_tokens = encoded.len();
        }
    };
    let boxed = encoded.into_boxed_slice();
    Box::into_raw(boxed) as *mut usize
}

#[no_mangle]
pub extern "C" fn tiktoken_corebpe_encode_with_special_tokens(
    ptr: *mut CoreBPE,
    text: *const c_char,
    num_tokens: *mut usize,
) -> *mut usize {
    if ptr.is_null() {
        warn!("Null pointer provided for CoreBPE!");
        return std::ptr::null_mut();
    }
    if text.is_null() {
        warn!("Null pointer provided for text!");
        return std::ptr::null_mut();
    }
    let text = unsafe {
        let raw = CStr::from_ptr(text);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                warn!("Invalid UTF-8 sequence provided for text!");
                return std::ptr::null_mut();
            }
        }
    };
    let corebpe = unsafe { &mut *ptr };
    let encoded = corebpe.encode_with_special_tokens(text);
    unsafe {
        if !num_tokens.is_null() {
            *num_tokens = encoded.len();
        }
    };
    let boxed = encoded.into_boxed_slice();
    Box::into_raw(boxed) as *mut usize
}

#[no_mangle]
pub extern "C" fn tiktoken_corebpe_decode(
    ptr: *mut CoreBPE,
    tokens: *const usize,
    num_tokens: usize,
) -> *mut c_char {
    if ptr.is_null() {
        warn!("Null pointer provided for CoreBPE!");
        return std::ptr::null_mut();
    }
    if tokens.is_null() {
        warn!("Null pointer provided for tokens!");
        return std::ptr::null_mut();
    }
    let tokens = unsafe { std::slice::from_raw_parts(tokens, num_tokens) };
    let tokens: Vec<usize> = tokens.iter().map(|&x| x as usize).collect();

    let corebpe = unsafe { &mut *ptr };
    let decoded = corebpe.decode(tokens);
    let decoded = match decoded {
        Ok(decoded) => decoded,
        Err(_) => {
            warn!("Failed to decode!");
            return std::ptr::null_mut();
        }
    };
    let c_str = match std::ffi::CString::new(decoded) {
        Ok(c_str) => c_str,
        Err(_) => {
            warn!("Failed to convert to CString!");
            return std::ptr::null_mut();
        }
    };
    let ptr = c_str.into_raw();
    ptr
}

#[no_mangle]
pub extern "C" fn tiktoken_c_version() -> *const c_char {
    static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_tiktoken_c_version() {
        let version = tiktoken_c_version();
        let version = get_string_from_c_char(version).unwrap();
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }

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

    #[test]
    fn test_get_completion_max_tokens() {
        let model = CString::new("gpt-4").unwrap();
        let prompt = CString::new("I am a cat.").unwrap();
        let max_tokens = tiktoken_get_completion_max_tokens(model.as_ptr(), prompt.as_ptr());
        assert_eq!(max_tokens, 8187);
    }

    #[test]
    fn test_get_completion_max_tokens_invalid_model() {
        let model = CString::new("cat-gpt").unwrap();
        let prompt = CString::new("I am a cat.").unwrap();
        let max_tokens = tiktoken_get_completion_max_tokens(model.as_ptr(), prompt.as_ptr());
        assert_eq!(max_tokens, usize::MAX);
    }

    #[test]
    fn test_get_completion_max_tokens_null_model() {
        let prompt = CString::new("I am a cat.").unwrap();
        let max_tokens = tiktoken_get_completion_max_tokens(std::ptr::null(), prompt.as_ptr());
        assert_eq!(max_tokens, usize::MAX);
    }

    #[test]
    fn test_get_completion_max_tokens_null_prompt() {
        let model = CString::new("gpt-4").unwrap();
        let max_tokens = tiktoken_get_completion_max_tokens(model.as_ptr(), std::ptr::null());
        assert_eq!(max_tokens, usize::MAX);
    }

    #[test]
    fn test_num_tokens_from_messages() {
        let model = CString::new("gpt-4").unwrap();
        let role = CString::new("system").unwrap();
        let content = CString::new("I am a cat.").unwrap();
        let message = CChatCompletionRequestMessage {
            role: role.as_ptr(),
            content: content.as_ptr(),
            name: std::ptr::null(),
            function_call: std::ptr::null(),
        };
        let messages = vec![message];
        let num_tokens = tiktoken_num_tokens_from_messages(
            model.as_ptr(),
            messages.len() as u32,
            messages.as_ptr(),
        );
        assert_eq!(num_tokens, 12);
    }

    #[test]
    fn test_num_tokens_from_messages_null_model() {
        let role = CString::new("system").unwrap();
        let content = CString::new("I am a cat.").unwrap();
        let message = CChatCompletionRequestMessage {
            role: role.as_ptr(),
            content: content.as_ptr(),
            name: std::ptr::null(),
            function_call: std::ptr::null(),
        };
        let messages = vec![message];
        let num_tokens = tiktoken_num_tokens_from_messages(
            std::ptr::null(),
            messages.len() as u32,
            messages.as_ptr(),
        );
        assert_eq!(num_tokens, usize::MAX);
    }

    #[test]
    fn test_num_tokens_from_messages_null_messages() {
        let model = CString::new("gpt-4").unwrap();
        let num_tokens = tiktoken_num_tokens_from_messages(model.as_ptr(), 0, std::ptr::null());
        assert_eq!(num_tokens, usize::MAX);
    }

    #[test]
    fn test_get_chat_completion_max_tokens() {
        let model = CString::new("gpt-4").unwrap();
        let role = CString::new("system").unwrap();
        let content = CString::new("I am a cat.").unwrap();
        let message = CChatCompletionRequestMessage {
            role: role.as_ptr(),
            content: content.as_ptr(),
            name: std::ptr::null(),
            function_call: std::ptr::null(),
        };
        let messages = vec![message];
        let max_tokens = tiktoken_get_chat_completion_max_tokens(
            model.as_ptr(),
            messages.len() as u32,
            messages.as_ptr(),
        );
        assert_eq!(max_tokens, 8180);
    }

    #[test]
    fn test_get_chat_completion_max_tokens_null_model() {
        let role = CString::new("system").unwrap();
        let content = CString::new("I am a cat.").unwrap();
        let message = CChatCompletionRequestMessage {
            role: role.as_ptr(),
            content: content.as_ptr(),
            name: std::ptr::null(),
            function_call: std::ptr::null(),
        };
        let messages = vec![message];
        let max_tokens = tiktoken_get_chat_completion_max_tokens(
            std::ptr::null(),
            messages.len() as u32,
            messages.as_ptr(),
        );
        assert_eq!(max_tokens, usize::MAX);
    }

    #[test]
    fn test_get_chat_completion_max_tokens_invalid_model() {
        let model = CString::new("cat-gpt").unwrap();
        let role = CString::new("system").unwrap();
        let content = CString::new("I am a cat.").unwrap();
        let message = CChatCompletionRequestMessage {
            role: role.as_ptr(),
            content: content.as_ptr(),
            name: std::ptr::null(),
            function_call: std::ptr::null(),
        };
        let messages = vec![message];
        let max_tokens = tiktoken_get_chat_completion_max_tokens(
            model.as_ptr(),
            messages.len() as u32,
            messages.as_ptr(),
        );
        assert_eq!(max_tokens, usize::MAX);
    }

    #[test]
    fn test_get_chat_completion_max_tokens_null_role() {
        let model = CString::new("gpt-4").unwrap();
        let content = CString::new("I am a cat.").unwrap();
        let message = CChatCompletionRequestMessage {
            role: std::ptr::null(),
            content: content.as_ptr(),
            name: std::ptr::null(),
            function_call: std::ptr::null(),
        };
        let messages = vec![message];
        let max_tokens = tiktoken_get_chat_completion_max_tokens(
            model.as_ptr(),
            messages.len() as u32,
            messages.as_ptr(),
        );
        assert_eq!(max_tokens, usize::MAX);
    }

    #[test]
    fn test_corebpe_encode_ordinary() {
        let model = CString::new("gpt-4").unwrap();
        let text = CString::new("I am a cat.").unwrap();
        let mut num_tokens: usize = 0;
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let tokens = tiktoken_corebpe_encode_ordinary(corebpe, text.as_ptr(), &mut num_tokens);
        assert_eq!(num_tokens, 5);
        let tokens = unsafe { std::slice::from_raw_parts(tokens, num_tokens) };
        let tokens: Vec<usize> = tokens.iter().map(|&x| x as usize).collect();
        assert_eq!(tokens, vec![40, 1097, 264, 8415, 13]);
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_corebpe_encode_ordinary_with_special_tokens() {
        let model = CString::new("gpt-4").unwrap();
        let text = CString::new("I am a cat. <|endoftext|>").unwrap();
        let mut num_tokens: usize = 0;
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let tokens =
            tiktoken_corebpe_encode_with_special_tokens(corebpe, text.as_ptr(), &mut num_tokens);
        assert_eq!(num_tokens, 7);
        let tokens = unsafe { std::slice::from_raw_parts(tokens, num_tokens) };
        let tokens: Vec<usize> = tokens.iter().map(|&x| x as usize).collect();
        assert_eq!(tokens, vec![40, 1097, 264, 8415, 13, 220, 100257]);
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_corebpe_encode_ordinary_null_corebpe() {
        let text = CString::new("I am a cat.").unwrap();
        let mut num_tokens: usize = 0;
        let tokens =
            tiktoken_corebpe_encode_ordinary(std::ptr::null_mut(), text.as_ptr(), &mut num_tokens);
        assert!(tokens.is_null());
    }

    #[test]
    fn test_corebpe_encode_ordinary_null_text() {
        let model = CString::new("gpt-4").unwrap();
        let mut num_tokens: usize = 0;
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let tokens = tiktoken_corebpe_encode_ordinary(corebpe, std::ptr::null(), &mut num_tokens);
        assert!(tokens.is_null());
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_crebpe_encode() {
        let model = CString::new("gpt-4").unwrap();
        let text = CString::new("I am a cat. <|fim_prefix|><|endoftext|>").unwrap();
        let mut num_tokens: usize = 0;
        let allowed_special: Vec<String> =
            vec!["<|endoftext|>".to_string(), "<|fim_prefix|>".to_string()];
        let allowed_special: Vec<CString> = allowed_special
            .iter()
            .map(|x| CString::new(x.as_str()).unwrap())
            .collect();
        let allowed_special: Vec<*const c_char> =
            allowed_special.iter().map(|x| x.as_ptr()).collect();
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let tokens = tiktoken_corebpe_encode(
            corebpe,
            text.as_ptr(),
            allowed_special.as_ptr(),
            allowed_special.len(),
            &mut num_tokens,
        );
        assert_eq!(num_tokens, 8);
        let tokens = unsafe { std::slice::from_raw_parts(tokens, num_tokens) };
        let tokens: Vec<usize> = tokens.iter().map(|&x| x as usize).collect();
        assert_eq!(tokens, vec![40, 1097, 264, 8415, 13, 220, 100258, 100257]);
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_crebpe_encode_null_corebpe() {
        let text = CString::new("I am a cat. <|endoftext|>").unwrap();
        let mut num_tokens: usize = 0;
        let allowed_special: Vec<String> = vec!["<|endoftext|>".to_string()];
        let allowed_special: Vec<CString> = allowed_special
            .iter()
            .map(|x| CString::new(x.as_str()).unwrap())
            .collect();
        let allowed_special: Vec<*const c_char> =
            allowed_special.iter().map(|x| x.as_ptr()).collect();
        let corebpe = std::ptr::null_mut();
        let tokens = tiktoken_corebpe_encode(
            corebpe,
            text.as_ptr(),
            allowed_special.as_ptr(),
            allowed_special.len(),
            &mut num_tokens,
        );
        assert!(tokens.is_null());
    }

    #[test]
    fn test_crebpe_encode_null_text() {
        let model = CString::new("gpt-4").unwrap();
        let text = std::ptr::null();
        let mut num_tokens: usize = 0;
        let allowed_special: Vec<String> = vec!["<|endoftext|>".to_string()];
        let allowed_special: Vec<CString> = allowed_special
            .iter()
            .map(|x| CString::new(x.as_str()).unwrap())
            .collect();
        let allowed_special: Vec<*const c_char> =
            allowed_special.iter().map(|x| x.as_ptr()).collect();
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let tokens = tiktoken_corebpe_encode(
            corebpe,
            text,
            allowed_special.as_ptr(),
            allowed_special.len(),
            &mut num_tokens,
        );
        assert!(tokens.is_null());
    }

    #[test]
    fn test_crebpe_encode_without_special_tokens() {
        let model = CString::new("gpt-4").unwrap();
        let text = CString::new("I am a cat. <|endoftext|>").unwrap();
        let mut num_tokens: usize = 0;
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());

        // zero-length slices require a non-null pointer
        // Create a CString and an array of pointers to pass to the function
        let placeholder = CString::new("").unwrap();
        let placeholder_ptr: *const i8 = placeholder.as_ptr();
        let ptr_array: [*const i8; 1] = [placeholder_ptr];

        let tokens = tiktoken_corebpe_encode(
            corebpe,
            text.as_ptr(),
            ptr_array.as_ptr(),
            0,
            &mut num_tokens,
        );
        assert_eq!(num_tokens, 11);
        let tokens = unsafe { std::slice::from_raw_parts(tokens, num_tokens) };
        let tokens: Vec<usize> = tokens.iter().map(|&x| x as usize).collect();
        assert_eq!(
            tokens,
            vec![40, 1097, 264, 8415, 13, 83739, 8862, 728, 428, 91, 29]
        );
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_corebpe_encode_with_special_tokens() {
        let model = CString::new("gpt-4").unwrap();
        let text = CString::new("I am a cat.").unwrap();
        let mut num_tokens: usize = 0;
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let tokens =
            tiktoken_corebpe_encode_with_special_tokens(corebpe, text.as_ptr(), &mut num_tokens);
        assert_eq!(num_tokens, 5);
        let tokens = unsafe { std::slice::from_raw_parts(tokens, num_tokens) };
        let tokens: Vec<usize> = tokens.iter().map(|&x| x as usize).collect();
        assert_eq!(tokens, vec![40, 1097, 264, 8415, 13]);
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_corebpe_encode_with_special_tokens_with_special_tokens() {
        let model = CString::new("gpt-4").unwrap();
        let text = CString::new("I am a cat. <|endoftext|>").unwrap();
        let mut num_tokens: usize = 0;
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let tokens =
            tiktoken_corebpe_encode_with_special_tokens(corebpe, text.as_ptr(), &mut num_tokens);
        assert_eq!(num_tokens, 7);
        let tokens = unsafe { std::slice::from_raw_parts(tokens, num_tokens) };
        let tokens: Vec<usize> = tokens.iter().map(|&x| x as usize).collect();
        assert_eq!(tokens, vec![40, 1097, 264, 8415, 13, 220, 100257]);
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_corebpe_encode_with_special_tokens_null_corebpe() {
        let text = CString::new("I am a cat.").unwrap();
        let mut num_tokens: usize = 0;
        let tokens = tiktoken_corebpe_encode_with_special_tokens(
            std::ptr::null_mut(),
            text.as_ptr(),
            &mut num_tokens,
        );
        assert!(tokens.is_null());
    }

    #[test]
    fn test_corebpe_encode_with_special_tokens_null_text() {
        let model = CString::new("gpt-4").unwrap();
        let mut num_tokens: usize = 0;
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let tokens =
            tiktoken_corebpe_encode_with_special_tokens(corebpe, std::ptr::null(), &mut num_tokens);
        assert!(tokens.is_null());
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_corebpe_decode() {
        let model = CString::new("gpt-4").unwrap();
        let tokens = vec![40, 1097, 264, 8415, 13];
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let decoded = tiktoken_corebpe_decode(corebpe, tokens.as_ptr(), tokens.len());
        let decoded = unsafe { CStr::from_ptr(decoded) };
        let decoded = decoded.to_str().unwrap();
        assert_eq!(decoded, "I am a cat.");
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_corebpe_decode_null_corebpe() {
        let tokens = vec![40, 1097, 264, 8415, 13];
        let decoded = tiktoken_corebpe_decode(std::ptr::null_mut(), tokens.as_ptr(), tokens.len());
        assert!(decoded.is_null());
    }

    #[test]
    fn test_corebpe_decode_null_tokens() {
        let model = CString::new("gpt-4").unwrap();
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let decoded = tiktoken_corebpe_decode(corebpe, std::ptr::null(), 0);
        assert!(decoded.is_null());
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_corebpe_decode_invalid_tokens() {
        let model = CString::new("gpt-4").unwrap();
        let tokens = vec![40, 1097, 264, 8415, 13, 220, 100257];
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let decoded = tiktoken_corebpe_decode(corebpe, tokens.as_ptr(), tokens.len());
        let decoded = unsafe { CStr::from_ptr(decoded) };
        let decoded = decoded.to_str().unwrap();
        assert_eq!(decoded, "I am a cat. <|endoftext|>");
        tiktoken_destroy_corebpe(corebpe);
    }
}
