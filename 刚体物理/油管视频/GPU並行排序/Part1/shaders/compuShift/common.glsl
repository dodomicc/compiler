float decode_float(vec4 pixel_data) {
    // 每个通道是 0~1 的 float，先转成 0~255 的整数
    ivec4 rgba = ivec4(pixel_data * 255.0);

    // 组合成 32-bit 整数
    uint packed = (uint(rgba[3]) << 24) |
                  (uint(rgba[2]) << 16) |
                  (uint(rgba[1]) << 8)  |
                  uint(rgba[0]);

    // 使用 uintBitsToFloat 解码为 float
    return uintBitsToFloat(packed);
}

float get_param_by_id(sampler2D sample,float id){
    float row = floor(id/128.);
    float col = mod(id,128.);
    ivec2 ip = ivec2(int(col),int(row));
    vec4 pixel_data = texelFetch(sample, ip, 0);
    return decode_float(pixel_data);
}

