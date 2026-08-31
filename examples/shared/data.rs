use bytemuck::Pod;
use std::mem::size_of_val;

pub fn byte_len<T>(values: &[T]) -> Result<u64, String> {
    u64::try_from(size_of_val(values)).map_err(|_| "value byte size exceeds u64".to_owned())
}

pub fn slice_bytes<T: Pod>(values: &[T]) -> &[u8] {
    bytemuck::cast_slice(values)
}

pub fn bytes_of<T: Pod>(value: &T) -> &[u8] {
    bytemuck::bytes_of(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn assert_pod<T: Pod>() {}

    #[test]
    fn pod_byte_views_cover_empty_slices_and_exact_value_storage() {
        assert_pod::<u16>();
        assert_pod::<[f32; 4]>();
        assert!(slice_bytes::<u32>(&[]).is_empty());
        assert_eq!(byte_len(&[1_u32, 2]), Ok(8));
        assert_eq!(slice_bytes(&[0x0102_u16]), 0x0102_u16.to_ne_bytes());
        assert_eq!(bytes_of(&0x0102_u16), 0x0102_u16.to_ne_bytes());
    }
}
