//这个函数用于输出如下结构。比如序列 1，1，2，5，5，11，11，输出一个等长数列，这个数列每个位置表示在该元素及以前有多少个不等元素，这里结果为1，1，2，3，3，4，4
const float merge_k = 256.;
//这个函数是用于一般比较在iChannel1中两个元素是否相等，可以进行自定义配置
bool is_equal(float idx1, float idx2){
    float num1 = get_float_by_particle_id_offset(iChannel1,idx1,0.);
    float num2 = get_float_by_particle_id_offset(iChannel1,idx2,0.);
    return num1 == num2;
}

float update_prefix_sum_by_idxs(float particle_id){

    float frame = get_frame();
    //这里作为前缀和的开始，从第1帧开始排序
    float group_size = pow(merge_k, frame + 1.);
    float internal_group_size = pow(merge_k, frame );
    float group_start = group_size * floor(particle_id/group_size);
    float group_id = floor(particle_id/group_size);
    float internal__group_id = floor(mod(particle_id,group_size)/internal_group_size);
    float res = get_float_by_particle_id_offset(iChannel0,particle_id,0.); 
    for (int i = 1; i<int(merge_k); i++){
        if(float(i)>internal__group_id) break;
        float last_internal_group_end = group_start + float(i) * internal_group_size - 1.;
        bool local_equal_res = is_equal(last_internal_group_end , last_internal_group_end  + 1.);
        res += get_float_by_particle_id_offset(iChannel0,last_internal_group_end,0.);
        res += local_equal_res ? (-1.) : 0.;

    }
    return res;             
}


float get_init_height(){
    return get_height(iChannel1);
}

float get_init_width(){
    return 1.;
}

float get_init_particle_data(float particle_id, float offset){
    //初始化

    float interval_id = floor(particle_id/merge_k);
    float interval_offset = mod(particle_id,merge_k);
    float interval_start = interval_id * merge_k;
    float res = 1.; 
    for (int i = 1; i<int(merge_k); i++){
        if(float(i)>interval_offset) break;
        res += is_equal(interval_start + float(i) - 1., interval_start + float(i) )?0.:1.;
    }
    return res;
}

float update_height(){
    return get_height(iChannel1);
}

float update_width(){
     return 1.;
}

float update_particle(float particle_id, float offset){
//因为该粒子偏移始终为1，因而直接输出本次循环中该粒子对应的prefix sum
    return update_prefix_sum_by_idxs(particle_id);
}