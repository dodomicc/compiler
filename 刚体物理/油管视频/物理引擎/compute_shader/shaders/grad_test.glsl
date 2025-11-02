



float get_init_height(){
    return get_height(iChannel1);
}

float get_init_width(){
    
    return get_width(iChannel1);
}

float get_init_particle_data(float particle_id, float offset){
    float[arr_len] jac = x_cross_y_jac(vec3(0.12,0.49,-0.31),vec3(0.12,-0.76,0.50));
    //  float[arr_len] jac = x_norm_jac(vec3(0.12,0.49,-0.31));
    // float[arr_len] jac = xi_minus_xj_jac(0,2,4);
    return get_float_by_particle_id_offset(iChannel1,particle_id,offset);

    return get_num_from_mat(jac,int(particle_id),int(offset)) ;
}

float update_height(){
    return 1.;
}

float update_width(){
     return 3.;
}

float update_particle(float particle_id, float offset){
//因为该粒子偏移始终为1，因而直接输出本次循环中该粒子对应的prefix sum
    return offset;
}