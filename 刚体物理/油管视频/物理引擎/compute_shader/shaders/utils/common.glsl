//元素位置转换
float read_float_from_texture(sampler2D tex,float id){
    float pixel_id = floor(id/4.);
    float pixel_row = floor(pixel_id/iResolution.x);
    float pixel_col = mod(pixel_id ,iResolution.x);
    vec2 uv = (vec2(pixel_col,pixel_row) + 0.5)/iResolution.xy;
    vec4 pixel = texelFetch(tex,ivec2(int(pixel_col),int(pixel_row)),0).xyzw;
    float type = mod(id,4.);
    if(type<0.5){
        return pixel.x;
    }else if(type<1.5){
          return pixel.y;
    }else if(type<2.5){
          return pixel.z;
    }else{
          return pixel.w;
    }
}

float get_height(sampler2D tex){
      return read_float_from_texture(tex,0.);
}

float get_width(sampler2D tex){
      return read_float_from_texture(tex,1.);
}

float get_id_by_fragcoord_and_offset(vec2 fragCoord, float offset){
      vec2 coords  = floor(fragCoord.xy);
      float id = coords.x + coords.y * iResolution.x;
      return 4. * id + offset;
}

bool is_height(vec2 fragCoord, float offset){
      float id = get_id_by_fragcoord_and_offset(fragCoord,offset);
      return abs(id)<0.5 ? true: false;
}

bool is_width(vec2 fragCoord, float offset){
      float id = get_id_by_fragcoord_and_offset(fragCoord,offset);
      return abs(id - 1.)<0.5? true: false;
}

bool is_particle(sampler2D tex, vec2 fragCoord, float offset){
      float height = read_float_from_texture(tex,0.);
      float width = read_float_from_texture(tex,1.);
      float id = get_id_by_fragcoord_and_offset(fragCoord,offset);
      return ((id >1.5) && (id<2. + height * width));
}

float get_partcle_id(sampler2D tex, vec2 fragCoord, float offset){
      float height = read_float_from_texture(tex,0.);
      float width = read_float_from_texture(tex,1.);
      float id = get_id_by_fragcoord_and_offset(fragCoord,offset)-2.;
      return floor(id/width);
}

float get_partcle_offset(sampler2D tex, vec2 fragCoord, float offset){
      float height = read_float_from_texture(tex,0.);
      float width = read_float_from_texture(tex,1.);
      float id = get_id_by_fragcoord_and_offset(fragCoord,offset) - 2.;
      return mod(id,width);
}

float get_float_by_particle_id_offset(sampler2D tex, float particle_id, float offset){
      float height = read_float_from_texture(tex,0.);
      float width = read_float_from_texture(tex,1.);
      return read_float_from_texture(tex,2. + width * particle_id + offset);
}

float get_frame(){
    return float(iFrame);
}
