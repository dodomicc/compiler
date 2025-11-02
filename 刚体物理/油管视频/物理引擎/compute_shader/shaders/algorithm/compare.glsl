const float merge_k = 8.;


// -------- 核心比较算子,可以外部自定义 ---------------------
float compare(float idx1, float idx2)
{
    float num1 = get_float_by_particle_id_offset(iChannel1,idx1, 0.);
    float num2 = get_float_by_particle_id_offset(iChannel1,idx2, 0.);
    return num1 - num2;            // >0 说明 num2 > num1
}

// -------- 计数工具：区间内 < idx 的元素数量 -----------------------------
float find_less_num_count(float start, float end, float idx)
{
    if (compare(get_float_by_particle_id_offset(iChannel0,start,0.), 
    get_float_by_particle_id_offset(iChannel0,idx,0.)) >= 0.0) 
        return 0;
    if (compare(get_float_by_particle_id_offset(iChannel0,end,0.),   
    get_float_by_particle_id_offset(iChannel0,idx,0.)) <  0.0) 
        return end - start + 1.0;
    float s = start;
    float e = end;
    // 二分最多 32 次即可覆盖 4G 数据
    for (int it = 0; it < 32; ++it) {
        if (!(e - s > 1.5)) continue;
        float mid = floor((s + e) * 0.5);
        if (compare(
                get_float_by_particle_id_offset(iChannel0,mid,0.), 
                get_float_by_particle_id_offset(iChannel0,idx,0.)
            ) < 0.0)
            s = mid;
        else
            e = mid;
    }
    return s - start + 1.0;
}

// 区间内 <= idx 的数量
float find_less_equal_num_count(float start, float end, float idx)
{
    if (compare(get_float_by_particle_id_offset(iChannel0,start,0.), 
    get_float_by_particle_id_offset(iChannel0,idx,0.)) >  0.0) 
        return 0;
    if (compare(get_float_by_particle_id_offset(iChannel0,end,0.),   
    get_float_by_particle_id_offset(iChannel0,idx,0.)) <= 0.0) 
        return end - start + 1.0;
    float s = start;
    float e = end;
    for (int it = 0; it < 32; ++it) {
        if (!(e - s > 1.5)) break;
        float mid = floor((s + e) * 0.5);
        if (compare(
                get_float_by_particle_id_offset(iChannel0,mid,0.), 
                get_float_by_particle_id_offset(iChannel0,idx,0.)
            ) <= 0.0)
            s = mid;
        else
            e = mid;
    }
    return s - start + 1.0;
}

float get_global_sorted_rank_in_group(float start, float end, float pointer_idx){
    float frames = get_frame();
    float group_size = pow(merge_k,frames + 1.);
    float sub_group_size = pow(merge_k,frames);
    float size = get_height(iChannel1);
    float res = start;
    float internal_idx = mod(pointer_idx - start,sub_group_size);
    float sub_interval_id = floor((pointer_idx - start)/sub_group_size);
    for (float i = 0.; i<merge_k; i++){
        float sub_start = start + i * sub_group_size;
        if(sub_start>=size - 1.) break;
        float sub_end = min(start + (i+1.) * sub_group_size - 1., size - 1.);
        if(i<sub_interval_id){
            res += find_less_equal_num_count(sub_start,sub_end,pointer_idx);
        }else if(i == sub_interval_id){
            res += internal_idx;
        }else{
            res += find_less_num_count(sub_start,sub_end,pointer_idx);
        }
    }
    return res;
}

float find_local_index_by_sorted_rank(float start, float end, float sub_interval_offset, float idx){
    float frames = get_frame();
    float group_size = pow(merge_k,frames + 1.);
    float size = get_height(iChannel1);
    float sub_group_size = pow(merge_k,frames);
    float sub_start = start + sub_group_size * sub_interval_offset;
    float sub_end = min(start + sub_group_size * (sub_interval_offset + 1.)- 1.,end);
    if(get_global_sorted_rank_in_group(start,end,sub_start)>idx) return -1.;
    if(get_global_sorted_rank_in_group(start,end,sub_end)<idx) return -1.;
    if(get_global_sorted_rank_in_group(start,end,sub_start) == idx) return sub_start;
    if(get_global_sorted_rank_in_group(start,end,sub_end) == idx) return sub_end;
    float s = sub_start;
    float e = sub_end;
    for(int i = 0; i<32; i++){
        if(e - s<1.5) break;
        float mid  = floor((s+ e)/2.);
        float idx0 = get_global_sorted_rank_in_group(start,end,mid);
        if(idx0 < idx){
            s = mid;
        }else if(idx0 == idx){
            return mid;
        }else{
            e = mid;
        }
    }
    if(get_global_sorted_rank_in_group(start,end,s) == idx) return s;
    if(get_global_sorted_rank_in_group(start,end,e) == idx) return e;
    return -1.;
}


// -------- 顶层：更新某元素 ---------------------------------------------
float update_sortd_idx(float particle_id)
{

    float frames            = get_frame();
    float group_size        = pow(merge_k,frames + 1.);
    float sub_group_size    = pow(merge_k,frames);
    float size              = get_height(iChannel1); 
    float start             = floor(particle_id/group_size) * group_size;
    float end               = min(start + group_size-1.,size - 1.);
    if(end - start + 1. <= sub_group_size) return get_float_by_particle_id_offset(iChannel0,particle_id,0.);
    for(float i = 0.; i<merge_k; i++){
        if((start + i * sub_group_size)>end) break;
        float perm = find_local_index_by_sorted_rank(start,end,i,particle_id);
        if(perm>-0.5) return get_float_by_particle_id_offset(iChannel0,perm,0.);
    }
    return 0.;
}



float get_init_height(){
    return get_height(iChannel1);
}

float get_init_width(){
    return 1.;
}



float get_init_particle_data(float particle_id, float offset){
    //初始化
    float group_size = merge_k;
    float group_id = floor(particle_id/group_size);
    float start = group_id * group_size;
    float size = get_height(iChannel1);
    float end = min(start + group_size - 1., size - 1.);
    float group_offset = mod(particle_id,merge_k);
    float[int(merge_k)] sorted_idx;


    for(float i = start; i<end + 1.; i++) sorted_idx[int(i - start)] = i;
    
    for(float i = end; i>=start; i--){
        for(float j = start; j<i ; j++){
            if(compare(sorted_idx[int(j-start)],sorted_idx[int(j+1 - start)])>0.){
                float temp = sorted_idx[int(j+1 - start)];
                sorted_idx[int(j+1 - start)] = sorted_idx[int(j-start)];
                sorted_idx[int(j-start)] = temp;
            }
        }
    }


  
    
    
    return sorted_idx[int(group_offset)];
}

float update_height(){
    return get_height(iChannel1);
}

float update_width(){
     return 1.;
}

float update_particle(float particle_id, float offset){
    return update_sortd_idx(particle_id);
}