void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float frame = float(iFrame);
    fragColor = vec4(0.);
    if(frame< 0.5){
         for(float i=0.; i<4.; i++){
            float id = get_id_by_fragcoord_and_offset(fragCoord,i);
            if(is_height(fragCoord,i)) {
                fragColor[int(i)] = get_init_height();
            }else if(is_width(fragCoord,i)) {
                fragColor[int(i)] = get_init_width();
            }else if(id<(2. + get_init_height() * get_init_width())){
                float particle_id = floor((id - 2.)/get_init_width());
                float offset0 = mod(id - 2.,get_init_width());
                fragColor[int(i)] = get_init_particle_data(particle_id,offset0);
            }else{
                return;
            }
        }
    }else{
        for(float i=0.; i<4.; i++){
            if(is_height(fragCoord,i)) 
                fragColor[int(i)] = update_height();
            if(is_width(fragCoord,i)) 
                fragColor[int(i)] = update_width();
            if(is_particle(iChannel0,fragCoord,i))
                fragColor[int(i)] = update_particle(
                    get_partcle_id(iChannel0,fragCoord,i),
                    get_partcle_offset(iChannel0,fragCoord,i)
                );
        }
    }
    return ;   
}