const float merge_k = 256.;
float compare(float idx1,float idx2){
    float num1 = get_float_by_particle_id_offset(iChannel1,idx1,0.);
    float num2 = get_float_by_particle_id_offset(iChannel1,idx2,0.);
    return num1 - num2;
}


float get_candidate_particle_idx(float particle_id){
    return get_float_by_particle_id_offset(iChannel0,particle_id,0.);
}


float max_val_idx(float particle_id) {
    float size = get_height(iChannel0);
    float start_idx = particle_id * merge_k;
    float end_idx = (particle_id + 1.0) * merge_k;
    if (start_idx < size + 0.5) {
        float res = get_candidate_particle_idx(start_idx);
        end_idx = min(size, end_idx);
        for(float i = 0.; i<merge_k; i++){
            // 如果 start_idx + i + 1.0这个位置超过当前  end_idx - 1， 就退出循环
            if((start_idx + i + 1.0) > (end_idx - 1. + 0.5)) break;
            float cand = get_candidate_particle_idx(start_idx + i + 1.0);
            float compare_res = compare(res, cand);
            if (compare_res > 0.0) res = cand;
        }
        return res;
    } else {
        // 该位置不需要计算排序，直接返回候选值
        return get_candidate_particle_idx(particle_id);
    }
}


float get_init_height(){
    return ceil(get_height(iChannel1)/merge_k);
}

float get_init_width(){
    return 1.;
}

float get_init_particle_data(float particle_id, float offset){
    float start_idx = merge_k * particle_id;
    float end_idx = merge_k * (particle_id + 1.) - 1.;
    end_idx = min(end_idx,get_height(iChannel1)-1.);
    float res = start_idx;
    for(float i = 1.; i<=merge_k; i++){
        if(start_idx + i > end_idx) break;
        if(compare(res, start_idx + i)>0.0) res = start_idx + i;
    }
    return res;
}

float update_height(){
    return ceil(get_height(iChannel0)/merge_k);
}

float update_width(){
     return 1.;
}

float update_particle(float particle_id, float offset){
    return max_val_idx(particle_id);
}