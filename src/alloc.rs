use std::ffi::CString;
use std::os::raw::{c_char, c_void};

/// Copy a slice into a malloc-allocated buffer and return the pointer.
/// Returns null on allocation failure or when len == 0.
pub fn malloc_copy<T: Copy>(slice: &[T]) -> *mut T {
    use std::mem::size_of;
    let len = slice.len();
    if len == 0 {
        return std::ptr::null_mut();
    }
    let bytes = match len.checked_mul(size_of::<T>()) {
        Some(b) if b > 0 => b,
        _ => return std::ptr::null_mut(),
    };
    let ptr = unsafe { libc::malloc(bytes) as *mut T };
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        std::ptr::copy_nonoverlapping(slice.as_ptr(), ptr, len);
    }
    ptr
}

/// Allocate with malloc and copy CString contents (including NUL).
/// Returns null on allocation failure.
pub fn cstring_into_malloced(c: CString) -> *mut c_char {
    let bytes = c.as_bytes_with_nul();
    let ptr = unsafe { libc::malloc(bytes.len()) as *mut c_char };
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        std::ptr::copy_nonoverlapping(c.as_ptr(), ptr, bytes.len());
    }
    ptr
}

#[no_mangle]
pub extern "C" fn tiktoken_free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    unsafe { libc::free(ptr) }
}
