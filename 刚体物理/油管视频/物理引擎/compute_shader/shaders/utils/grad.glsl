float[arr_len] xi_minus_xj_jac(int i, int j, int n){
    float[200] jac = create_zeros_mat(3.,3. * float(n));
    for (int k = 0; k<3; k++){
        set_num_for_mat(jac,k, 3 * i + k, 1.);
        set_num_for_mat(jac,k, 3 * j + k, -1.);
    }
    return jac;
}

float[arr_len] xi_add_xj_jac(int i, int j, int n){
    float[200] jac = create_zeros_mat(3.,3. * float(n));
    for (int k = 0; k<3; k++){
        set_num_for_mat(jac, k, 3 * i + k, 1.);
        set_num_for_mat(jac, k, 3 * j + k, 1.);
    }
    return jac;
}

float[arr_len] x_cross_y_jac(vec3 x, vec3 y){
    float[arr_len] jac = create_zeros_mat(3.,6.);
    set_num_for_mat(jac,0,1,y[2]);
    set_num_for_mat(jac,0,2,-y[1]);
    set_num_for_mat(jac,1,0,-y[2]);
    set_num_for_mat(jac,1,2,y[0]);
    set_num_for_mat(jac,2,0,y[1]);
    set_num_for_mat(jac,2,1,-y[0]);

    set_num_for_mat(jac,0,4,-x[2]);
    set_num_for_mat(jac,0,5,x[1]);
    set_num_for_mat(jac,1,3,x[2]);
    set_num_for_mat(jac,1,5,-x[0]);
    set_num_for_mat(jac,2,3,-x[1]);
    set_num_for_mat(jac,2,4,x[0]);

    return jac;
}

float[arr_len] x_norm_jac(vec3 x){
    x = x/(length(x) + 1e-6);
    float[arr_len] x1 = turn_vec3_into_mat(x);
    as_row_vec(x1);
    return x1;
}

float[arr_len] x_normalize_jac(vec3 x){
    float norm = length(x) + 1e-6;
    mat3 identity = mat3(1.);
    mat3 jac = 1./norm * (identity - outer_product(x/norm,x/norm));
    return turn_mat3_into_mat(jac);
}

float[arr_len] x_y_dot_jac(vec3 x, vec3 y){
    float[arr_len] jac;
    jac[0] = 1.;
    jac[1] = 6.;
    jac[2] = y[0];
    jac[3] = y[1];
    jac[4] = y[2];
    jac[5] = x[0];
    jac[6] = x[1];
    jac[7] = x[2];
    return jac;
}