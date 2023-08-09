use std::ffi::CStr;
use std::os::raw::c_char;
use tiktoken_rs;
use tiktoken_rs::CoreBPE;

#[no_mangle]
pub extern "C" fn c_r50k_base() -> *mut CoreBPE {
    let bpe = tiktoken_rs::r50k_base();
    let corebpe = bpe.unwrap();
    let boxed = Box::new(corebpe);
    Box::into_raw(boxed)
}

#[no_mangle]
pub extern "C" fn c_p50k_base() -> *mut CoreBPE {
    let bpe = tiktoken_rs::p50k_base();
    let corebpe = bpe.unwrap();
    let boxed = Box::new(corebpe);
    Box::into_raw(boxed)
}

#[no_mangle]
pub extern "C" fn c_p50k_edit() -> *mut CoreBPE {
    let bpe = tiktoken_rs::p50k_edit();
    let corebpe = bpe.unwrap();
    let boxed = Box::new(corebpe);
    Box::into_raw(boxed)
}

#[no_mangle]
pub extern "C" fn c_cl100k_base() -> *mut CoreBPE {
    let bpe = tiktoken_rs::cl100k_base();
    let corebpe = bpe.unwrap();
    let boxed = Box::new(corebpe);
    Box::into_raw(boxed)
}

#[no_mangle]
pub extern "C" fn c_destroy_corebpe(ptr: *mut CoreBPE) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(ptr);
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
            eprintln!("Invalid UTF-8 sequence provided!");
            return None;
        }
    };

    Some(c_str)
}

#[no_mangle]
pub extern "C" fn c_get_completion_max_tokens(
    model: *const c_char,
    prompt: *const c_char,
) -> usize {
    if model.is_null() {
        eprintln!("Null pointer provided for model!");
        return usize::MAX;
    }
    if prompt.is_null() {
        eprintln!("Null pointer provided for prompt!");
        return usize::MAX;
    }
    let model = unsafe {
        let raw = CStr::from_ptr(model);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                eprintln!("Invalid UTF-8 sequence provided for model!");
                return usize::MAX;
            }
        }
    };
    let prompt = unsafe {
        let raw = CStr::from_ptr(prompt);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                eprintln!("Invalid UTF-8 sequence provided for prompt!");
                return usize::MAX;
            }
        }
    };
    match tiktoken_rs::get_completion_max_tokens(model, prompt) {
        Ok(max_tokens) => max_tokens,
        Err(_) => {
            eprintln!("Failed to get completion max tokens!");
            return usize::MAX;
        }
    }
}

#[no_mangle]
pub extern "C" fn c_num_tokens_from_messages(
    model: *const c_char,
    num_messages: u32,
    messages: *const CChatCompletionRequestMessage,
) -> usize {
    if model.is_null() {
        eprintln!("Null pointer provided for model!");
        return usize::MAX;
    }
    if messages.is_null() {
        eprintln!("Null pointer provided for messages!");
        return usize::MAX;
    }
    let model = unsafe {
        let raw = CStr::from_ptr(model);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                eprintln!("Invalid UTF-8 sequence provided for model!");
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
            eprintln!("Failed to get num tokens!");
            return usize::MAX;
        }
    }
}

#[no_mangle]
pub extern "C" fn c_get_chat_completion_max_tokens(
    model: *const c_char,
    num_messages: u32,
    messages: *const CChatCompletionRequestMessage,
) -> usize {
    if model.is_null() {
        eprintln!("Null pointer provided for model!");
        return usize::MAX;
    }
    if messages.is_null() {
        eprintln!("Null pointer provided for messages!");
        return usize::MAX;
    }
    let model = unsafe {
        let raw = CStr::from_ptr(model);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                eprintln!("Invalid UTF-8 sequence provided for model!");
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
    match tiktoken_rs::get_chat_completion_max_tokens(model, &messages) {
        Ok(max_tokens) => max_tokens,
        Err(_) => {
            eprintln!("Failed to get max tokens!");
            return usize::MAX;
        }
    }
}

#[no_mangle]
pub extern "C" fn c_get_bpe_from_model(model: *const c_char) -> *mut CoreBPE {
    if model.is_null() {
        eprintln!("Null pointer provided for model!");
        return std::ptr::null_mut();
    }
    let model = unsafe {
        let raw = CStr::from_ptr(model);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                eprintln!("Invalid UTF-8 sequence provided for model!");
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
            eprintln!("Failed to get BPE from model!");
            std::ptr::null_mut()
        }
    }
}

// get_bpe_from_tokenizer is not yet implemented.
// Use c_r50k_base(), c_p50k_base(), c_p50k_edit(), and c_cl100k_base()
// instead.

#[no_mangle]
pub extern "C" fn c_corebpe_encode_ordinary(
    ptr: *mut CoreBPE,
    text: *const c_char,
    num_tokens: *mut usize,
) -> *mut usize {
    if ptr.is_null() {
        eprintln!("Null pointer provided for CoreBPE!");
        return std::ptr::null_mut();
    }
    if text.is_null() {
        eprintln!("Null pointer provided for text!");
        return std::ptr::null_mut();
    }
    let text = unsafe {
        let raw = CStr::from_ptr(text);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                eprintln!("Invalid UTF-8 sequence provided for text!");
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

#[no_mangle]
pub extern "C" fn c_corebpe_encode_with_special_tokens(
    ptr: *mut CoreBPE,
    text: *const c_char,
    num_tokens: *mut usize,
) -> *mut usize {
    if ptr.is_null() {
        eprintln!("Null pointer provided for CoreBPE!");
        return std::ptr::null_mut();
    }
    if text.is_null() {
        eprintln!("Null pointer provided for text!");
        return std::ptr::null_mut();
    }
    let text = unsafe {
        let raw = CStr::from_ptr(text);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                eprintln!("Invalid UTF-8 sequence provided for text!");
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
pub extern "C" fn c_corebpe_decode(
    ptr: *mut CoreBPE,
    tokens: *const usize,
    num_tokens: usize,
) -> *mut c_char {
    if ptr.is_null() {
        eprintln!("Null pointer provided for CoreBPE!");
        return std::ptr::null_mut();
    }
    if tokens.is_null() {
        eprintln!("Null pointer provided for tokens!");
        return std::ptr::null_mut();
    }
    let tokens = unsafe { std::slice::from_raw_parts(tokens, num_tokens) };
    let tokens: Vec<usize> = tokens.iter().map(|&x| x as usize).collect();

    let corebpe = unsafe { &mut *ptr };
    let decoded = corebpe.decode(tokens);
    let decoded = match decoded {
        Ok(decoded) => decoded,
        Err(_) => {
            eprintln!("Failed to decode!");
            return std::ptr::null_mut();
        }
    };
    let c_str = match std::ffi::CString::new(decoded) {
        Ok(c_str) => c_str,
        Err(_) => {
            eprintln!("Failed to convert to CString!");
            return std::ptr::null_mut();
        }
    };
    let ptr = c_str.into_raw();
    ptr
}
