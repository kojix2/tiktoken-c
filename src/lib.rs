#[cfg(feature = "logging")]
use log::warn;
#[cfg(feature = "logging")]
use simple_logger::SimpleLogger;
use std::ffi::{c_char, CStr};
use tiktoken_rs;
use tiktoken_rs::{CoreBPE, Rank};

mod alloc;
use alloc::{cstring_into_malloced, malloc_copy};

mod corebpe;
// use corebpe::{
//     tiktoken_cl100k_base, tiktoken_destroy_corebpe, tiktoken_get_bpe_from_model,
//     tiktoken_o200k_base, tiktoken_p50k_base, tiktoken_p50k_edit, tiktoken_r50k_base,
// };

mod utils;
use utils::c_str_to_string;

#[cfg(feature = "logging")]
#[no_mangle]
pub extern "C" fn tiktoken_init_logger() {
    SimpleLogger::new().init().unwrap();
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CTiktokenTokenizer {
    Unknown = 0,
    O200kHarmony = 1,
    O200kBase = 2,
    Cl100kBase = 3,
    P50kBase = 4,
    R50kBase = 5,
    P50kEdit = 6,
    Gpt2 = 7,
}

impl From<tiktoken_rs::tokenizer::Tokenizer> for CTiktokenTokenizer {
    fn from(tokenizer: tiktoken_rs::tokenizer::Tokenizer) -> Self {
        match tokenizer {
            tiktoken_rs::tokenizer::Tokenizer::O200kHarmony => Self::O200kHarmony,
            tiktoken_rs::tokenizer::Tokenizer::O200kBase => Self::O200kBase,
            tiktoken_rs::tokenizer::Tokenizer::Cl100kBase => Self::Cl100kBase,
            tiktoken_rs::tokenizer::Tokenizer::P50kBase => Self::P50kBase,
            tiktoken_rs::tokenizer::Tokenizer::R50kBase => Self::R50kBase,
            tiktoken_rs::tokenizer::Tokenizer::P50kEdit => Self::P50kEdit,
            tiktoken_rs::tokenizer::Tokenizer::Gpt2 => Self::Gpt2,
        }
    }
}

#[no_mangle]
pub extern "C" fn tiktoken_get_context_size(model: *const c_char) -> usize {
    let model = match parse_required_string(model, "model") {
        Ok(model) => model,
        Err(_) => return usize::MAX,
    };

    match tiktoken_rs::model::get_context_size(&model) {
        Some(context_size) => context_size,
        None => {
            #[cfg(feature = "logging")]
            warn!("Failed to get context size!");
            usize::MAX
        }
    }
}

#[no_mangle]
pub extern "C" fn tiktoken_get_tokenizer(model: *const c_char) -> CTiktokenTokenizer {
    let model = match parse_required_string(model, "model") {
        Ok(model) => model,
        Err(_) => return CTiktokenTokenizer::Unknown,
    };

    match tiktoken_rs::tokenizer::get_tokenizer(&model) {
        Some(tokenizer) => tokenizer.into(),
        None => {
            #[cfg(feature = "logging")]
            warn!("Failed to get tokenizer!");
            CTiktokenTokenizer::Unknown
        }
    }
}

#[no_mangle]
pub extern "C" fn tiktoken_get_text_completion_max_tokens(
    model: *const c_char,
    prompt: *const c_char,
) -> usize {
    if model.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for model!");
        return usize::MAX;
    }
    if prompt.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for prompt!");
        return usize::MAX;
    }
    let model = unsafe {
        let raw = CStr::from_ptr(model);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                #[cfg(feature = "logging")]
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
                #[cfg(feature = "logging")]
                warn!("Invalid UTF-8 sequence provided for prompt!");
                return usize::MAX;
            }
        }
    };
    match tiktoken_rs::get_text_completion_max_tokens(model, prompt) {
        Ok(max_tokens) => max_tokens,
        Err(_) => {
            #[cfg(feature = "logging")]
            warn!("Failed to get completion max tokens!");
            return usize::MAX;
        }
    }
}

#[repr(C)]
pub struct CChatCompletionRequestMessage {
    inner: tiktoken_rs::ChatCompletionRequestMessage,
}

fn parse_required_string(ptr: *const c_char, _field_name: &str) -> Result<String, ()> {
    if ptr.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for {}!", _field_name);
        return Err(());
    }
    match c_str_to_string(ptr) {
        Some(value) => Ok(value),
        None => {
            #[cfg(feature = "logging")]
            warn!("Invalid UTF-8 sequence provided for {}!", _field_name);
            Err(())
        }
    }
}

fn parse_optional_string(ptr: *const c_char, _field_name: &str) -> Result<Option<String>, ()> {
    if ptr.is_null() {
        return Ok(None);
    }
    match c_str_to_string(ptr) {
        Some(value) => Ok(Some(value)),
        None => {
            #[cfg(feature = "logging")]
            warn!("Invalid UTF-8 sequence provided for {}!", _field_name);
            Err(())
        }
    }
}

fn parse_function_call(
    name: *const c_char,
    arguments: *const c_char,
    field_name: &str,
) -> Result<tiktoken_rs::FunctionCall, ()> {
    Ok(tiktoken_rs::FunctionCall {
        name: parse_required_string(name, &format!("{} name", field_name))?,
        arguments: parse_required_string(arguments, &format!("{} arguments", field_name))?,
    })
}

fn parse_chat_messages(
    num_messages: u32,
    messages: *const *mut CChatCompletionRequestMessage,
) -> Result<Vec<tiktoken_rs::ChatCompletionRequestMessage>, ()> {
    let slice = unsafe { std::slice::from_raw_parts(messages, num_messages as usize) };
    let mut messages_vec = Vec::with_capacity(num_messages as usize);

    for &message in slice {
        if message.is_null() {
            #[cfg(feature = "logging")]
            warn!("Null pointer provided for message!");
            return Err(());
        }
        messages_vec.push(unsafe { (&*message).inner.clone() });
    }

    Ok(messages_vec)
}

#[no_mangle]
pub extern "C" fn tiktoken_chat_message_new(
    role: *const c_char,
) -> *mut CChatCompletionRequestMessage {
    let role = match parse_required_string(role, "role") {
        Ok(role) => role,
        Err(_) => return std::ptr::null_mut(),
    };
    Box::into_raw(Box::new(CChatCompletionRequestMessage {
        inner: tiktoken_rs::ChatCompletionRequestMessage {
            role,
            ..Default::default()
        },
    }))
}

#[no_mangle]
pub extern "C" fn tiktoken_chat_message_set_role(
    ptr: *mut CChatCompletionRequestMessage,
    role: *const c_char,
) -> bool {
    if ptr.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for message!");
        return false;
    }
    let role = match parse_required_string(role, "role") {
        Ok(role) => role,
        Err(_) => return false,
    };
    unsafe { &mut *ptr }.inner.role = role;
    true
}

#[no_mangle]
pub extern "C" fn tiktoken_chat_message_set_content(
    ptr: *mut CChatCompletionRequestMessage,
    content: *const c_char,
) -> bool {
    if ptr.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for message!");
        return false;
    }
    let content = match parse_optional_string(content, "content") {
        Ok(content) => content,
        Err(_) => return false,
    };
    unsafe { &mut *ptr }.inner.content = content;
    true
}

#[no_mangle]
pub extern "C" fn tiktoken_chat_message_set_name(
    ptr: *mut CChatCompletionRequestMessage,
    name: *const c_char,
) -> bool {
    if ptr.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for message!");
        return false;
    }
    let name = match parse_optional_string(name, "name") {
        Ok(name) => name,
        Err(_) => return false,
    };
    unsafe { &mut *ptr }.inner.name = name;
    true
}

#[no_mangle]
pub extern "C" fn tiktoken_chat_message_set_function_call(
    ptr: *mut CChatCompletionRequestMessage,
    name: *const c_char,
    arguments: *const c_char,
) -> bool {
    if ptr.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for message!");
        return false;
    }
    let function_call = match parse_function_call(name, arguments, "function_call") {
        Ok(function_call) => function_call,
        Err(_) => return false,
    };
    unsafe { &mut *ptr }.inner.function_call = Some(function_call);
    true
}

#[no_mangle]
pub extern "C" fn tiktoken_chat_message_clear_function_call(
    ptr: *mut CChatCompletionRequestMessage,
) {
    if ptr.is_null() {
        return;
    }
    unsafe { &mut *ptr }.inner.function_call = None;
}

#[no_mangle]
pub extern "C" fn tiktoken_chat_message_add_tool_call(
    ptr: *mut CChatCompletionRequestMessage,
    name: *const c_char,
    arguments: *const c_char,
) -> bool {
    if ptr.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for message!");
        return false;
    }
    let tool_call = match parse_function_call(name, arguments, "tool_call") {
        Ok(tool_call) => tool_call,
        Err(_) => return false,
    };
    unsafe { &mut *ptr }.inner.tool_calls.push(tool_call);
    true
}

#[no_mangle]
pub extern "C" fn tiktoken_chat_message_clear_tool_calls(
    ptr: *mut CChatCompletionRequestMessage,
) {
    if ptr.is_null() {
        return;
    }
    unsafe { &mut *ptr }.inner.tool_calls.clear();
}

#[no_mangle]
pub extern "C" fn tiktoken_chat_message_set_refusal(
    ptr: *mut CChatCompletionRequestMessage,
    refusal: *const c_char,
) -> bool {
    if ptr.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for message!");
        return false;
    }
    let refusal = match parse_optional_string(refusal, "refusal") {
        Ok(refusal) => refusal,
        Err(_) => return false,
    };
    unsafe { &mut *ptr }.inner.refusal = refusal;
    true
}

#[no_mangle]
pub extern "C" fn tiktoken_chat_message_destroy(ptr: *mut CChatCompletionRequestMessage) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(ptr);
    }
}

#[no_mangle]
pub extern "C" fn tiktoken_num_tokens_from_messages(
    model: *const c_char,
    num_messages: u32,
    messages: *const *mut CChatCompletionRequestMessage,
) -> usize {
    if model.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for model!");
        return usize::MAX;
    }
    if messages.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for messages!");
        return usize::MAX;
    }
    let model = unsafe {
        let raw = CStr::from_ptr(model);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                #[cfg(feature = "logging")]
                warn!("Invalid UTF-8 sequence provided for model!");
                return usize::MAX;
            }
        }
    };
    let messages = match parse_chat_messages(num_messages, messages) {
        Ok(messages) => messages,
        Err(_) => {
            return usize::MAX;
        }
    };
    match tiktoken_rs::num_tokens_from_messages(model, &messages) {
        Ok(num_tokens) => num_tokens,
        Err(_) => {
            #[cfg(feature = "logging")]
            warn!("Failed to get num tokens!");
            return usize::MAX;
        }
    }
}

#[no_mangle]
pub extern "C" fn tiktoken_get_chat_completion_max_tokens(
    model: *const c_char,
    num_messages: u32,
    messages: *const *mut CChatCompletionRequestMessage,
) -> usize {
    if model.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for model!");
        return usize::MAX;
    }
    if messages.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for messages!");
        return usize::MAX;
    }
    let model = unsafe {
        let raw = CStr::from_ptr(model);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                #[cfg(feature = "logging")]
                warn!("Invalid UTF-8 sequence provided for model!");
                return usize::MAX;
            }
        }
    };
    let messages = match parse_chat_messages(num_messages, messages) {
        Ok(messages) => messages,
        Err(_) => {
            return usize::MAX;
        }
    };
    match tiktoken_rs::get_chat_completion_max_tokens(model, &messages) {
        Ok(max_tokens) => max_tokens,
        Err(_) => {
            #[cfg(feature = "logging")]
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
) -> *mut Rank {
    if ptr.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for CoreBPE!");
        return std::ptr::null_mut();
    }
    if text.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for text!");
        return std::ptr::null_mut();
    }
    let text = unsafe {
        let raw = CStr::from_ptr(text);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                #[cfg(feature = "logging")]
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
    malloc_copy::<Rank>(&encoded)
}

#[no_mangle]
pub extern "C" fn tiktoken_corebpe_count_ordinary(ptr: *mut CoreBPE, text: *const c_char) -> usize {
    if ptr.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for CoreBPE!");
        return usize::MAX;
    }
    if text.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for text!");
        return usize::MAX;
    }
    let text = unsafe {
        let raw = CStr::from_ptr(text);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                #[cfg(feature = "logging")]
                warn!("Invalid UTF-8 sequence provided for text!");
                return usize::MAX;
            }
        }
    };
    let corebpe = unsafe { &mut *ptr };
    corebpe.count_ordinary(text)
}

// pub fn encode(&self, text: &str, allowed_special: HashSet<&str>) -> Vec<usize>
#[no_mangle]
pub extern "C" fn tiktoken_corebpe_encode(
    ptr: *mut CoreBPE,
    text: *const c_char,
    allowed_special: *const *const c_char,
    allowed_special_len: usize,
    num_tokens: *mut usize,
) -> *mut Rank {
    if ptr.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for CoreBPE!");
        return std::ptr::null_mut();
    }
    if text.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for text!");
        return std::ptr::null_mut();
    }
    if allowed_special.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for allowed_special!");
        return std::ptr::null_mut();
    }
    let text = unsafe {
        let raw = CStr::from_ptr(text);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                #[cfg(feature = "logging")]
                warn!("Invalid UTF-8 sequence provided for text!");
                return std::ptr::null_mut();
            }
        }
    };
    let allowed_special = unsafe {
        let slice = std::slice::from_raw_parts(allowed_special, allowed_special_len);
        let mut allowed_special_hash_set = std::collections::HashSet::new();
        for i in 0..allowed_special_len {
            if slice[i].is_null() {
                #[cfg(feature = "logging")]
                warn!("Null pointer provided in allowed_special!");
                return std::ptr::null_mut();
            }
            let c_str = CStr::from_ptr(slice[i]);
            let str_slice = match c_str.to_str() {
                Ok(valid_str) => valid_str,
                Err(_) => {
                    #[cfg(feature = "logging")]
                    warn!("Invalid UTF-8 sequence provided for allowed_special!");
                    return std::ptr::null_mut();
                }
            };
            allowed_special_hash_set.insert(str_slice);
        }
        allowed_special_hash_set
    };
    let corebpe = unsafe { &mut *ptr };
    let encoded = corebpe.encode(text, &allowed_special).0;
    unsafe {
        if !num_tokens.is_null() {
            *num_tokens = encoded.len();
        }
    };
    malloc_copy::<Rank>(&encoded)
}

#[no_mangle]
pub extern "C" fn tiktoken_corebpe_count(
    ptr: *mut CoreBPE,
    text: *const c_char,
    allowed_special: *const *const c_char,
    allowed_special_len: usize,
) -> usize {
    if ptr.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for CoreBPE!");
        return usize::MAX;
    }
    if text.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for text!");
        return usize::MAX;
    }
    let text = unsafe {
        let raw = CStr::from_ptr(text);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                #[cfg(feature = "logging")]
                warn!("Invalid UTF-8 sequence provided for text!");
                return usize::MAX;
            }
        }
    };
    let allowed_special = unsafe {
        if allowed_special_len == 0 {
            std::collections::HashSet::new()
        } else {
            if allowed_special.is_null() {
                #[cfg(feature = "logging")]
                warn!("Null pointer provided for allowed_special!");
                return usize::MAX;
            }
            let slice = std::slice::from_raw_parts(allowed_special, allowed_special_len);
            let mut allowed_special_hash_set = std::collections::HashSet::new();
            for item in slice {
                if item.is_null() {
                    #[cfg(feature = "logging")]
                    warn!("Null pointer provided in allowed_special!");
                    return usize::MAX;
                }
                let c_str = CStr::from_ptr(*item);
                let str_slice = match c_str.to_str() {
                    Ok(valid_str) => valid_str,
                    Err(_) => {
                        #[cfg(feature = "logging")]
                        warn!("Invalid UTF-8 sequence provided for allowed_special!");
                        return usize::MAX;
                    }
                };
                allowed_special_hash_set.insert(str_slice);
            }
            allowed_special_hash_set
        }
    };
    let corebpe = unsafe { &mut *ptr };
    corebpe.count(text, &allowed_special)
}

#[no_mangle]
pub extern "C" fn tiktoken_corebpe_encode_with_special_tokens(
    ptr: *mut CoreBPE,
    text: *const c_char,
    num_tokens: *mut usize,
) -> *mut Rank {
    if ptr.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for CoreBPE!");
        return std::ptr::null_mut();
    }
    if text.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for text!");
        return std::ptr::null_mut();
    }
    let text = unsafe {
        let raw = CStr::from_ptr(text);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                #[cfg(feature = "logging")]
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
    malloc_copy::<Rank>(&encoded)
}

#[no_mangle]
pub extern "C" fn tiktoken_corebpe_count_with_special_tokens(
    ptr: *mut CoreBPE,
    text: *const c_char,
) -> usize {
    if ptr.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for CoreBPE!");
        return usize::MAX;
    }
    if text.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for text!");
        return usize::MAX;
    }
    let text = unsafe {
        let raw = CStr::from_ptr(text);
        match raw.to_str() {
            Ok(valid_str) => valid_str,
            Err(_) => {
                #[cfg(feature = "logging")]
                warn!("Invalid UTF-8 sequence provided for text!");
                return usize::MAX;
            }
        }
    };
    let corebpe = unsafe { &mut *ptr };
    corebpe.count_with_special_tokens(text)
}

#[no_mangle]
pub extern "C" fn tiktoken_corebpe_decode(
    ptr: *mut CoreBPE,
    tokens: *const Rank,
    num_tokens: usize,
) -> *mut c_char {
    if ptr.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for CoreBPE!");
        return std::ptr::null_mut();
    }
    if tokens.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for tokens!");
        return std::ptr::null_mut();
    }
    let tokens = unsafe { std::slice::from_raw_parts(tokens, num_tokens) };

    let corebpe = unsafe { &mut *ptr };
    let decoded = corebpe.decode(tokens);
    let decoded = match decoded {
        Ok(decoded) => decoded,
        Err(_) => {
            #[cfg(feature = "logging")]
            warn!("Failed to decode!");
            return std::ptr::null_mut();
        }
    };
    let c_str = match std::ffi::CString::new(decoded) {
        Ok(c_str) => c_str,
        Err(_) => {
            #[cfg(feature = "logging")]
            warn!("Failed to convert to CString!");
            return std::ptr::null_mut();
        }
    };
    cstring_into_malloced(c_str)
}

#[no_mangle]
pub extern "C" fn tiktoken_corebpe_decode_bytes(
    ptr: *mut CoreBPE,
    tokens: *const Rank,
    num_tokens: usize,
    num_bytes: *mut usize,
) -> *mut u8 {
    if ptr.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for CoreBPE!");
        return std::ptr::null_mut();
    }
    if tokens.is_null() {
        #[cfg(feature = "logging")]
        warn!("Null pointer provided for tokens!");
        return std::ptr::null_mut();
    }
    let tokens = unsafe { std::slice::from_raw_parts(tokens, num_tokens) };

    let corebpe = unsafe { &mut *ptr };
    let decoded = corebpe.decode_bytes(tokens);
    let decoded = match decoded {
        Ok(decoded) => decoded,
        Err(_) => {
            #[cfg(feature = "logging")]
            warn!("Failed to decode bytes!");
            return std::ptr::null_mut();
        }
    };
    unsafe {
        if !num_bytes.is_null() {
            *num_bytes = decoded.len();
        }
    };
    malloc_copy::<u8>(&decoded)
}

#[no_mangle]
pub extern "C" fn tiktoken_c_version() -> *const c_char {
    static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::tiktoken_free;
    use corebpe::{tiktoken_destroy_corebpe, tiktoken_get_bpe_from_model, tiktoken_r50k_base};
    use std::ffi::CString;
    use utils::get_string_from_c_char;

    #[test]
    fn test_tiktoken_c_version() {
        let version = tiktoken_c_version();
        let version = get_string_from_c_char(version).unwrap();
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_get_context_size() {
        let model = CString::new("gpt-4o").unwrap();
        let context_size = tiktoken_get_context_size(model.as_ptr());
        assert_eq!(context_size, 128000);
    }

    #[test]
    fn test_get_context_size_ft_model() {
        let model = CString::new("ft:gpt-4o:org:model:id").unwrap();
        let context_size = tiktoken_get_context_size(model.as_ptr());
        assert_eq!(context_size, 128000);
    }

    #[test]
    fn test_get_context_size_invalid_model() {
        let model = CString::new("cat-gpt").unwrap();
        let context_size = tiktoken_get_context_size(model.as_ptr());
        assert_eq!(context_size, usize::MAX);
    }

    #[test]
    fn test_get_context_size_null_model() {
        let context_size = tiktoken_get_context_size(std::ptr::null());
        assert_eq!(context_size, usize::MAX);
    }

    #[test]
    fn test_get_tokenizer() {
        let model = CString::new("gpt-4o").unwrap();
        let tokenizer = tiktoken_get_tokenizer(model.as_ptr());
        assert_eq!(tokenizer, CTiktokenTokenizer::O200kBase);
    }

    #[test]
    fn test_get_tokenizer_gpt2() {
        let model = CString::new("gpt2").unwrap();
        let tokenizer = tiktoken_get_tokenizer(model.as_ptr());
        assert_eq!(tokenizer, CTiktokenTokenizer::Gpt2);
    }

    #[test]
    fn test_get_tokenizer_ft_model() {
        let model = CString::new("ft:gpt-3.5-turbo:org:model:id").unwrap();
        let tokenizer = tiktoken_get_tokenizer(model.as_ptr());
        assert_eq!(tokenizer, CTiktokenTokenizer::Cl100kBase);
    }

    #[test]
    fn test_get_tokenizer_invalid_model() {
        let model = CString::new("cat-gpt").unwrap();
        let tokenizer = tiktoken_get_tokenizer(model.as_ptr());
        assert_eq!(tokenizer, CTiktokenTokenizer::Unknown);
    }

    #[test]
    fn test_get_tokenizer_null_model() {
        let tokenizer = tiktoken_get_tokenizer(std::ptr::null());
        assert_eq!(tokenizer, CTiktokenTokenizer::Unknown);
    }

    #[test]
    fn test_get_text_completion_max_tokens() {
        let model = CString::new("gpt-4").unwrap();
        let prompt = CString::new("I am a cat.").unwrap();
        let max_tokens = tiktoken_get_text_completion_max_tokens(model.as_ptr(), prompt.as_ptr());
        assert_eq!(max_tokens, 8187);
    }

    #[test]
    fn test_get_text_completion_max_tokens_invalid_model() {
        let model = CString::new("cat-gpt").unwrap();
        let prompt = CString::new("I am a cat.").unwrap();
        let max_tokens = tiktoken_get_text_completion_max_tokens(model.as_ptr(), prompt.as_ptr());
        assert_eq!(max_tokens, usize::MAX);
    }

    #[test]
    fn test_get_text_completion_max_tokens_null_model() {
        let prompt = CString::new("I am a cat.").unwrap();
        let max_tokens =
            tiktoken_get_text_completion_max_tokens(std::ptr::null(), prompt.as_ptr());
        assert_eq!(max_tokens, usize::MAX);
    }

    #[test]
    fn test_get_text_completion_max_tokens_null_prompt() {
        let model = CString::new("gpt-4").unwrap();
        let max_tokens = tiktoken_get_text_completion_max_tokens(model.as_ptr(), std::ptr::null());
        assert_eq!(max_tokens, usize::MAX);
    }

    #[test]
    fn test_get_text_completion_max_tokens_gpt41() {
        // gpt-4.1 context size: 1,047,576; "I am a cat." encodes to 5 tokens (o200k)
        let model = CString::new("gpt-4.1").unwrap();
        let prompt = CString::new("I am a cat.").unwrap();
        let max_tokens = tiktoken_get_text_completion_max_tokens(model.as_ptr(), prompt.as_ptr());
        assert_eq!(max_tokens, 1_047_571);
    }

    #[test]
    fn test_get_text_completion_max_tokens_gpt5() {
        // gpt-5 context size: 400,000; "I am a cat." encodes to 5 tokens (o200k)
        let model = CString::new("gpt-5").unwrap();
        let prompt = CString::new("I am a cat.").unwrap();
        let max_tokens = tiktoken_get_text_completion_max_tokens(model.as_ptr(), prompt.as_ptr());
        assert_eq!(max_tokens, 399_995);
    }

    #[test]
    fn test_get_text_completion_max_tokens_basic_repeat() {
        let model = CString::new("gpt-4").unwrap();
        let prompt = CString::new("I am a cat.").unwrap();
        let max_tokens = tiktoken_get_text_completion_max_tokens(model.as_ptr(), prompt.as_ptr());
        assert_eq!(max_tokens, 8187);
    }

    fn message_array(
        messages: &[*mut CChatCompletionRequestMessage],
    ) -> Vec<*mut CChatCompletionRequestMessage> {
        messages.to_vec()
    }

    fn destroy_messages(messages: &[*mut CChatCompletionRequestMessage]) {
        for &message in messages {
            tiktoken_chat_message_destroy(message);
        }
    }

    #[test]
    fn test_chat_message_new_null_role() {
        let message = tiktoken_chat_message_new(std::ptr::null());
        assert!(message.is_null());
    }

    #[test]
    fn test_num_tokens_from_messages() {
        let model = CString::new("gpt-4").unwrap();
        let role = CString::new("system").unwrap();
        let content = CString::new("I am a cat.").unwrap();
        let message = tiktoken_chat_message_new(role.as_ptr());
        assert!(!message.is_null());
        assert!(tiktoken_chat_message_set_content(message, content.as_ptr()));
        let messages = [message];
        let message_ptrs = message_array(&messages);
        let num_tokens = tiktoken_num_tokens_from_messages(
            model.as_ptr(),
            message_ptrs.len() as u32,
            message_ptrs.as_ptr(),
        );
        assert_eq!(num_tokens, 12);
        destroy_messages(&messages);
    }

    #[test]
    fn test_num_tokens_from_messages_null_model() {
        let role = CString::new("system").unwrap();
        let content = CString::new("I am a cat.").unwrap();
        let message = tiktoken_chat_message_new(role.as_ptr());
        assert!(!message.is_null());
        assert!(tiktoken_chat_message_set_content(message, content.as_ptr()));
        let messages = [message];
        let message_ptrs = message_array(&messages);
        let num_tokens = tiktoken_num_tokens_from_messages(
            std::ptr::null(),
            message_ptrs.len() as u32,
            message_ptrs.as_ptr(),
        );
        assert_eq!(num_tokens, usize::MAX);
        destroy_messages(&messages);
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
        let message = tiktoken_chat_message_new(role.as_ptr());
        assert!(!message.is_null());
        assert!(tiktoken_chat_message_set_content(message, content.as_ptr()));
        let messages = [message];
        let message_ptrs = message_array(&messages);
        let max_tokens = tiktoken_get_chat_completion_max_tokens(
            model.as_ptr(),
            message_ptrs.len() as u32,
            message_ptrs.as_ptr(),
        );
        assert_eq!(max_tokens, 8180);
        destroy_messages(&messages);
    }

    #[test]
    fn test_get_chat_completion_max_tokens_null_model() {
        let role = CString::new("system").unwrap();
        let content = CString::new("I am a cat.").unwrap();
        let message = tiktoken_chat_message_new(role.as_ptr());
        assert!(!message.is_null());
        assert!(tiktoken_chat_message_set_content(message, content.as_ptr()));
        let messages = [message];
        let message_ptrs = message_array(&messages);
        let max_tokens = tiktoken_get_chat_completion_max_tokens(
            std::ptr::null(),
            message_ptrs.len() as u32,
            message_ptrs.as_ptr(),
        );
        assert_eq!(max_tokens, usize::MAX);
        destroy_messages(&messages);
    }

    #[test]
    fn test_get_chat_completion_max_tokens_invalid_model() {
        let model = CString::new("cat-gpt").unwrap();
        let role = CString::new("system").unwrap();
        let content = CString::new("I am a cat.").unwrap();
        let message = tiktoken_chat_message_new(role.as_ptr());
        assert!(!message.is_null());
        assert!(tiktoken_chat_message_set_content(message, content.as_ptr()));
        let messages = [message];
        let message_ptrs = message_array(&messages);
        let max_tokens = tiktoken_get_chat_completion_max_tokens(
            model.as_ptr(),
            message_ptrs.len() as u32,
            message_ptrs.as_ptr(),
        );
        assert_eq!(max_tokens, usize::MAX);
        destroy_messages(&messages);
    }

    #[test]
    fn test_get_chat_completion_max_tokens_null_message_item() {
        let model = CString::new("gpt-4").unwrap();
        let messages = [std::ptr::null_mut::<CChatCompletionRequestMessage>()];
        let max_tokens =
            tiktoken_get_chat_completion_max_tokens(model.as_ptr(), messages.len() as u32, messages.as_ptr());
        assert_eq!(max_tokens, usize::MAX);
    }

    #[test]
    fn test_num_tokens_from_messages_gpt41() {
        // gpt-4.1 uses O200kBase tokenizer; overhead is same as other current models
        let model = CString::new("gpt-4.1").unwrap();
        let role = CString::new("system").unwrap();
        let content = CString::new("I am a cat.").unwrap();
        let message = tiktoken_chat_message_new(role.as_ptr());
        assert!(!message.is_null());
        assert!(tiktoken_chat_message_set_content(message, content.as_ptr()));
        let messages = [message];
        let message_ptrs = message_array(&messages);
        let num_tokens = tiktoken_num_tokens_from_messages(
            model.as_ptr(),
            message_ptrs.len() as u32,
            message_ptrs.as_ptr(),
        );
        assert_eq!(num_tokens, 12);
        destroy_messages(&messages);
    }

    #[test]
    fn test_get_chat_completion_max_tokens_gpt41() {
        // gpt-4.1 context size: 1,047,576
        let model = CString::new("gpt-4.1").unwrap();
        let role = CString::new("system").unwrap();
        let content = CString::new("I am a cat.").unwrap();
        let message = tiktoken_chat_message_new(role.as_ptr());
        assert!(!message.is_null());
        assert!(tiktoken_chat_message_set_content(message, content.as_ptr()));
        let messages = [message];
        let message_ptrs = message_array(&messages);
        let max_tokens = tiktoken_get_chat_completion_max_tokens(
            model.as_ptr(),
            message_ptrs.len() as u32,
            message_ptrs.as_ptr(),
        );
        assert_eq!(max_tokens, 1_047_564);
        destroy_messages(&messages);
    }

    #[test]
    fn test_num_tokens_from_messages_gpt_oss() {
        // gpt-oss-20b uses O200kHarmony tokenizer, which is supported for chat counting
        let model = CString::new("gpt-oss-20b").unwrap();
        let role = CString::new("system").unwrap();
        let content = CString::new("I am a cat.").unwrap();
        let message = tiktoken_chat_message_new(role.as_ptr());
        assert!(!message.is_null());
        assert!(tiktoken_chat_message_set_content(message, content.as_ptr()));
        let messages = [message];
        let message_ptrs = message_array(&messages);
        let num_tokens = tiktoken_num_tokens_from_messages(
            model.as_ptr(),
            message_ptrs.len() as u32,
            message_ptrs.as_ptr(),
        );
        assert_eq!(num_tokens, 12);
        destroy_messages(&messages);
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
    fn test_corebpe_count_ordinary() {
        let model = CString::new("gpt-4").unwrap();
        let text = CString::new("I am a cat.").unwrap();
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let num_tokens = tiktoken_corebpe_count_ordinary(corebpe, text.as_ptr());
        assert_eq!(num_tokens, 5);
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
    fn test_corebpe_count_ordinary_null_corebpe() {
        let text = CString::new("I am a cat.").unwrap();
        let num_tokens = tiktoken_corebpe_count_ordinary(std::ptr::null_mut(), text.as_ptr());
        assert_eq!(num_tokens, usize::MAX);
    }

    #[test]
    fn test_corebpe_count_ordinary_null_text() {
        let model = CString::new("gpt-4").unwrap();
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let num_tokens = tiktoken_corebpe_count_ordinary(corebpe, std::ptr::null());
        assert_eq!(num_tokens, usize::MAX);
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
    fn test_crebpe_count() {
        let model = CString::new("gpt-4").unwrap();
        let text = CString::new("I am a cat. <|fim_prefix|><|endoftext|>").unwrap();
        let allowed_special: Vec<String> =
            vec!["<|endoftext|>".to_string(), "<|fim_prefix|>".to_string()];
        let allowed_special: Vec<CString> = allowed_special
            .iter()
            .map(|x| CString::new(x.as_str()).unwrap())
            .collect();
        let allowed_special: Vec<*const c_char> =
            allowed_special.iter().map(|x| x.as_ptr()).collect();
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let num_tokens = tiktoken_corebpe_count(
            corebpe,
            text.as_ptr(),
            allowed_special.as_ptr(),
            allowed_special.len(),
        );
        assert_eq!(num_tokens, 8);
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
    fn test_crebpe_encode_null_allowed_special_item() {
        let model = CString::new("gpt-4").unwrap();
        let text = CString::new("I am a cat.").unwrap();
        let mut num_tokens: usize = 0;
        let allowed_special: [*const c_char; 1] = [std::ptr::null()];
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let tokens = tiktoken_corebpe_encode(
            corebpe,
            text.as_ptr(),
            allowed_special.as_ptr(),
            allowed_special.len(),
            &mut num_tokens,
        );
        assert!(tokens.is_null());
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_crebpe_count_without_special_tokens() {
        let model = CString::new("gpt-4").unwrap();
        let text = CString::new("I am a cat. <|endoftext|>").unwrap();
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let num_tokens = tiktoken_corebpe_count(corebpe, text.as_ptr(), std::ptr::null(), 0);
        assert_eq!(num_tokens, 11);
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_crebpe_count_null_corebpe() {
        let text = CString::new("I am a cat. <|endoftext|>").unwrap();
        let num_tokens =
            tiktoken_corebpe_count(std::ptr::null_mut(), text.as_ptr(), std::ptr::null(), 0);
        assert_eq!(num_tokens, usize::MAX);
    }

    #[test]
    fn test_crebpe_count_null_text() {
        let model = CString::new("gpt-4").unwrap();
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let num_tokens = tiktoken_corebpe_count(corebpe, std::ptr::null(), std::ptr::null(), 0);
        assert_eq!(num_tokens, usize::MAX);
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_crebpe_count_null_allowed_special_item() {
        let model = CString::new("gpt-4").unwrap();
        let text = CString::new("I am a cat.").unwrap();
        let allowed_special: [*const c_char; 1] = [std::ptr::null()];
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let num_tokens = tiktoken_corebpe_count(
            corebpe,
            text.as_ptr(),
            allowed_special.as_ptr(),
            allowed_special.len(),
        );
        assert_eq!(num_tokens, usize::MAX);
        tiktoken_destroy_corebpe(corebpe);
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
    fn test_corebpe_count_with_special_tokens() {
        let model = CString::new("gpt-4").unwrap();
        let text = CString::new("I am a cat. <|endoftext|>").unwrap();
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let num_tokens = tiktoken_corebpe_count_with_special_tokens(corebpe, text.as_ptr());
        assert_eq!(num_tokens, 7);
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
    fn test_corebpe_count_with_special_tokens_null_corebpe() {
        let text = CString::new("I am a cat.").unwrap();
        let num_tokens =
            tiktoken_corebpe_count_with_special_tokens(std::ptr::null_mut(), text.as_ptr());
        assert_eq!(num_tokens, usize::MAX);
    }

    #[test]
    fn test_corebpe_count_with_special_tokens_null_text() {
        let model = CString::new("gpt-4").unwrap();
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let num_tokens = tiktoken_corebpe_count_with_special_tokens(corebpe, std::ptr::null());
        assert_eq!(num_tokens, usize::MAX);
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

    #[test]
    fn test_corebpe_decode_bytes() {
        let model = CString::new("gpt-4").unwrap();
        let tokens = vec![40, 1097, 264, 8415, 13];
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let mut num_bytes = 0;
        let decoded =
            tiktoken_corebpe_decode_bytes(corebpe, tokens.as_ptr(), tokens.len(), &mut num_bytes);
        let decoded = unsafe { std::slice::from_raw_parts(decoded, num_bytes) };
        assert_eq!(decoded, b"I am a cat.");
        tiktoken_free(decoded.as_ptr() as *mut libc::c_void);
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_corebpe_decode_bytes_non_utf8() {
        let corebpe = tiktoken_r50k_base();
        let tokens = [49426];
        let mut num_bytes = 0;
        let decoded =
            tiktoken_corebpe_decode_bytes(corebpe, tokens.as_ptr(), tokens.len(), &mut num_bytes);
        assert!(!decoded.is_null());
        assert!(num_bytes > 0);
        tiktoken_free(decoded as *mut libc::c_void);
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_corebpe_decode_bytes_null_corebpe() {
        let tokens = [40, 1097, 264, 8415, 13];
        let decoded = tiktoken_corebpe_decode_bytes(
            std::ptr::null_mut(),
            tokens.as_ptr(),
            tokens.len(),
            std::ptr::null_mut(),
        );
        assert!(decoded.is_null());
    }

    #[test]
    fn test_corebpe_decode_bytes_null_tokens() {
        let model = CString::new("gpt-4").unwrap();
        let corebpe = tiktoken_get_bpe_from_model(model.as_ptr());
        let decoded =
            tiktoken_corebpe_decode_bytes(corebpe, std::ptr::null(), 0, std::ptr::null_mut());
        assert!(decoded.is_null());
        tiktoken_destroy_corebpe(corebpe);
    }

    #[test]
    fn test_num_tokens_from_messages_with_function_call() {
        let model = CString::new("gpt-4").unwrap();
        let role = CString::new("assistant").unwrap();
        let content = CString::new("I'll help you with that.").unwrap();
        let fun_name = CString::new("get_weather").unwrap();
        let fun_args = CString::new("{\"location\": \"Tokyo\"}").unwrap();
        let message = tiktoken_chat_message_new(role.as_ptr());
        assert!(!message.is_null());
        assert!(tiktoken_chat_message_set_content(message, content.as_ptr()));
        assert!(tiktoken_chat_message_set_function_call(
            message,
            fun_name.as_ptr(),
            fun_args.as_ptr(),
        ));
        let messages = [message];
        let message_ptrs = message_array(&messages);
        let num_tokens = tiktoken_num_tokens_from_messages(
            model.as_ptr(),
            message_ptrs.len() as u32,
            message_ptrs.as_ptr(),
        );
        assert_ne!(num_tokens, usize::MAX);
        assert!(num_tokens > 0);
        destroy_messages(&messages);
    }

    #[test]
    fn test_num_tokens_from_messages_function_call_null_name() {
        let role = CString::new("assistant").unwrap();
        let fun_args = CString::new("{\"location\": \"Tokyo\"}").unwrap();
        let message = tiktoken_chat_message_new(role.as_ptr());
        assert!(!message.is_null());
        assert!(!tiktoken_chat_message_set_function_call(
            message,
            std::ptr::null(),
            fun_args.as_ptr(),
        ));
        tiktoken_chat_message_destroy(message);
    }

    #[test]
    fn test_get_chat_completion_max_tokens_with_function_call() {
        let model = CString::new("gpt-4").unwrap();
        let role = CString::new("assistant").unwrap();
        let content = CString::new("I'll help you with that.").unwrap();
        let fun_name = CString::new("get_weather").unwrap();
        let fun_args = CString::new("{\"location\": \"Tokyo\"}").unwrap();
        let message = tiktoken_chat_message_new(role.as_ptr());
        assert!(!message.is_null());
        assert!(tiktoken_chat_message_set_content(message, content.as_ptr()));
        assert!(tiktoken_chat_message_set_function_call(
            message,
            fun_name.as_ptr(),
            fun_args.as_ptr(),
        ));
        let messages = [message];
        let message_ptrs = message_array(&messages);
        let max_tokens = tiktoken_get_chat_completion_max_tokens(
            model.as_ptr(),
            message_ptrs.len() as u32,
            message_ptrs.as_ptr(),
        );
        assert_ne!(max_tokens, usize::MAX);
        assert!(max_tokens > 0);
        destroy_messages(&messages);
    }

    #[test]
    fn test_get_chat_completion_max_tokens_function_call_null_arguments() {
        let role = CString::new("assistant").unwrap();
        let fun_name = CString::new("get_weather").unwrap();
        let message = tiktoken_chat_message_new(role.as_ptr());
        assert!(!message.is_null());
        assert!(!tiktoken_chat_message_set_function_call(
            message,
            fun_name.as_ptr(),
            std::ptr::null(),
        ));
        tiktoken_chat_message_destroy(message);
    }

    #[test]
    fn test_num_tokens_from_messages_with_tool_calls() {
        let model = CString::new("gpt-4o").unwrap();
        let role = CString::new("assistant").unwrap();
        let content = CString::new("I'll call a tool.").unwrap();
        let tool_name = CString::new("get_weather").unwrap();
        let tool_args = CString::new("{\"location\":\"Tokyo\"}").unwrap();
        let message = tiktoken_chat_message_new(role.as_ptr());
        assert!(!message.is_null());
        assert!(tiktoken_chat_message_set_content(message, content.as_ptr()));
        assert!(tiktoken_chat_message_add_tool_call(
            message,
            tool_name.as_ptr(),
            tool_args.as_ptr(),
        ));
        let messages = [message];
        let message_ptrs = message_array(&messages);
        let num_tokens = tiktoken_num_tokens_from_messages(
            model.as_ptr(),
            message_ptrs.len() as u32,
            message_ptrs.as_ptr(),
        );
        assert_ne!(num_tokens, usize::MAX);
        assert!(num_tokens > 0);
        destroy_messages(&messages);
    }

    #[test]
    fn test_num_tokens_from_messages_with_refusal() {
        let model = CString::new("gpt-4o").unwrap();
        let role = CString::new("assistant").unwrap();
        let refusal = CString::new("I cannot help with that request.").unwrap();
        let message = tiktoken_chat_message_new(role.as_ptr());
        assert!(!message.is_null());
        assert!(tiktoken_chat_message_set_refusal(message, refusal.as_ptr()));
        let messages = [message];
        let message_ptrs = message_array(&messages);
        let num_tokens = tiktoken_num_tokens_from_messages(
            model.as_ptr(),
            message_ptrs.len() as u32,
            message_ptrs.as_ptr(),
        );
        assert_ne!(num_tokens, usize::MAX);
        assert!(num_tokens > 0);
        destroy_messages(&messages);
    }

    #[test]
    fn test_num_tokens_from_messages_tool_call_null_arguments() {
        let role = CString::new("assistant").unwrap();
        let tool_name = CString::new("get_weather").unwrap();
        let message = tiktoken_chat_message_new(role.as_ptr());
        assert!(!message.is_null());
        assert!(!tiktoken_chat_message_add_tool_call(
            message,
            tool_name.as_ptr(),
            std::ptr::null(),
        ));
        tiktoken_chat_message_destroy(message);
    }

    #[test]
    fn test_get_chat_completion_max_tokens_null_message_pointer() {
        let model = CString::new("gpt-4o").unwrap();
        let messages = [std::ptr::null_mut::<CChatCompletionRequestMessage>()];
        let max_tokens = tiktoken_get_chat_completion_max_tokens(
            model.as_ptr(),
            messages.len() as u32,
            messages.as_ptr(),
        );
        assert_eq!(max_tokens, usize::MAX);
    }
}
