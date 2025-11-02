float T_LINEAR_VELOCITY = 1.;
float L_LINEAR_VELOCITY = 3.;

float T_ANGULAR_VELOCITY = 2.;
float L_ANGULAR_VELOCITY = 3.;

float T_POS = 3.;
float L_POS = 3.;

float T_MASS = 4.;
float L_MASS = 1.;

struct particle_state{
    float particle_num;
    float single_particle_length;
    float linear_velocity_start;
    float angular_velocity_start;
    float pos_start;
    float mass_start;
}




vec4 encode_float(float value) {
    uint u = floatBitsToUint(value);  // 将 float 转成 32-bit uint

    // 提取每个字节（低位在前）
    float r = float(u & 0xFFu);
    float g = float((u >> 8) & 0xFFu);
    float b = float((u >> 16) & 0xFFu);
    float a = float((u >> 24) & 0xFFu);

    // 映射到 [0.0, 1.0]
    return vec4(r, g, b, a) / 255.0;
}


float decode_float(vec4 pixel_data){
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

float get_num_by_id(sampler2D sample,float id){
    float row = floor(id/128.);
    float col = mod(id,128.);
    vec2 ip = vec2(col + 0.5,row + 0.5)/iResolution.xy ;
    vec4 pixel_data = texture(sample, ip);
    return decode_float(pixel_data);
}

particle_state get_particle_state(sampler2D sample){
    float particle_num  = get_num_by_id(sample,0.);
    vec4 types = decode_float(get_num_by_id(sample,1.)) * 255.;
    particle_state ps;
    ps.linear_velocity_start = -1.;
    ps.angular_velocity_start = -1.;
    ps.mass_start = -1.;
    ps.pos_start = -1.;
    float curIdx  = 0.;
    for (int i = 0; i<4; i++){
        float type1 = floor(types[i]/16.);
        float type2 = mod(types[i],16.);
        if(abs(type1 - T_LINEAR_VELOCITY )<0.01 || abs(type2 - T_LINEAR_VELOCITY )<0.01){
                 ps.linear_velocity_start = curIdx;
                 curIdx += L_LINEAR_VELOCITY;
        }
        if(abs(type1 - T_ANGULAR_VELOCITY )<0.01 || abs(type2 - T_ANGULAR_VELOCITY) <0.01){
                 ps.angular_velocity_start = curIdx;
                 curIdx += L_ANGULAR_VELOCITY;
        }
        if(abs(type1 - T_POS)<0.01 || abs(type2 - T_POS)<0.01){
                 ps.pos_start = curIdx;
                 curIdx += L_POS;
        }
        if(abs(type1 - T_MASS)<0.01 || abs(type2 - T_MASS)<0.01){
                 ps.mass_start = curIdx;
                 curIdx += L_MASS;
        }
    }
    ps.particle_num = particle_num;
    ps.single_particle_length = curIdx;
    return ps;

}

vec3 get_type_and_offset_by_uv(vec2 uv, particle_state ps) {
    // 每行宽度固定为 128
    float width = 128.0;

    // 当前像素的 id
    vec2 fragCoord = uv * iResolution.xy;
    float row = floor(fragCoord.y);
    float col = floor(fragCoord.x);
    float id = row * width + col;

    // 跳过 metadata（前两个像素）
    if (id < 2.0) {
        return vec3(-1.0, -1.0, 0.0);
    }

    // 相对于粒子数据区的 index
    float local_id = id - 2.0;

    // 每粒子的字段长度
    float single_length = ps.single_particle_length;
    float particle_idx = floor(local_id / single_length);
    float offset_in_particle = mod(local_id, single_length);

    if (particle_idx >= ps.particle_num ) {
        return vec3(-1.0, -1.0, 0.0); // 超出粒子总数
    }


    // 线性速度
    if (
        ps.linear_velocity_start > -0.5 
        && offset_in_particle >= ps.linear_velocity_start 
        && offset_in_particle < ps.linear_velocity_start +  L_LINEAR_VELOCITY
    ) {
        return vec3(
            particle_idx, 
            T_LINEAR_VELOCITY, 
            offset_in_particle - ps.linear_velocity_start
            );
    }

    // 角速度

    if (
        ps.angular_velocity_start>-0.5 
        && offset_in_particle >= ps.angular_velocity_start 
        && offset_in_particle < ps.angular_velocity_start + L_ANGULAR_VELOCITY
    ) {
        return vec3(
            particle_idx,
            T_ANGULAR_VELOCITY, 
            offset_in_particle - ps.angular_velocity_start
        );
    }

    // 位置

    if (ps.pos_start >-0.5 
        && offset_in_particle >= ps.pos_start
        && offset_in_particle < ps.pos_start + L_POS) {
        return vec3(
            particle_idx,
            T_POS, 
            offset_in_particle - ps.pos_start
        );
    }

    // 质量

    if (ps.mass_start > -0.5 
        &&offset_in_particle >= ps.mass_start 
        && offset_in_particle < ps.mass_start + L_MASS) {
        return vec3(
            particle_idx,
            T_MASS, 
            offset_in_particle - ps.mass_start);
    }

    // 不属于任何已知字段
    return vec3(particle_idx, -1., 0.0);


}

vec3 read_vec3_field(sampler2D sample,float particle_idx,float start_idx, particle_state ps){
    float start_id = 2. + ps.single_particle_length * particle_idx + start_idx;
    return vec3(
        get_num_by_id(sample,start_id),
        get_num_by_id(sample,start_id + 1.),
        get_num_by_id(sample,start_id + 2.),
        )
}

vec3 read_vec2_field(sampler2D sample,float particle_idx,float start_idx, particle_state ps){
    float start_id = 2. + ps.single_particle_length * particle_idx + start_idx;
    return vec2(
        get_num_by_id(sample,start_id),
        get_num_by_id(sample,start_id + 1.)

        )
}

vec3 read_float_field(sampler2D sample,float particle_idx,float start_idx, particle_state ps){
    float start_id = 2. + ps.single_particle_length * particle_idx + start_idx;
    return get_num_by_id(sample,start_id)
}





