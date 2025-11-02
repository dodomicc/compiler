//矩阵部分
const int arr_len = 200;
const int horizontal_axis = 0;
const int vertical_axis = 1;
float[arr_len] d;

float[arr_len] create_zeros_mat(float height, float width){
      float[200] arr;
      arr[0] = height;
      arr[1] = width;
      for(int i =2; i<int(height * width) + 2; i++) arr[i] = 0.;
      
      return arr;
}

float[arr_len] create_ones_mat(float height, float width){
      float[200] arr;
      arr[0] = height;
      arr[1] = width;
      for (int i = 2; i<arr_len + 2; i++)
            arr[i] = 1.;
      return arr;
}

void set_num_for_mat(inout float[arr_len] mat, int row_idx, int col_idx, float num){
      float width = mat[1];
      float idx = row_idx * width + col_idx + 2.;
      mat[int(round(idx))] = num;

}

float get_num_from_mat(inout float[arr_len] mat, int row_idx, int col_idx){
      float width = mat[1];
      float idx = row_idx * width + col_idx + 2.;
      return mat[int(round(idx))];
}

void mat_scalar(inout float[arr_len] mat, float scalar){
      float height = mat[0];
      float width = mat[1];
      for (int i=0; i<2 + int(height * width); i++){
            mat[i]*=scalar;
      } 
}

float[arr_len] mat_add(inout float[arr_len] mat_1, inout float[arr_len] mat_2){
      float height = mat_1[0];
      float width = mat_1[1];
      float[arr_len] new_mat;
      for (int i=0; i<2 + int(height * width); i++){
            new_mat[i] = mat_1[i] + mat_2[i];
      } 
      return new_mat;
}

float[arr_len] mat_diff(inout float[arr_len] mat_1, inout float[arr_len] mat_2){
      float height = mat_1[0];
      float width = mat_1[1];
      float[arr_len] new_mat;
      for (int i=0; i<2 + int(height * width); i++){
            new_mat[i] = mat_1[i] - mat_2[i];
      } 
      return new_mat;
}

float[arr_len] mat_mul(float[arr_len] mat_1, float[arr_len] mat_2){
      float height = mat_1[0];
      float width = mat_2[1];
      float[arr_len] new_mat;
      for (int i= 0; i<height; i++){
            for (int j =0; j<width; j++){
                  float delta = 0.;
                  for(int k=0; k<mat_1[1]; k++){
                        delta += get_num_from_mat(mat_1,i,k) * get_num_from_mat(mat_2,k,j);
                  }
                   set_num_for_mat(new_mat,i,j,delta);
            }
      }
      return new_mat;

}

float[arr_len] mat_concentrate(float[arr_len] mat_1, float[arr_len] mat_2, int axis){
      float[arr_len] res;
      if(axis == horizontal_axis){
            res[0] = mat_1[0];
            res[1] = mat_1[1] + mat_2[1];
            int total_len = int(round(res[0]*res[1]));
            int height = int(round(res[0]));
            int width = int(round(res[1]));
            int left_width = int(round(mat_1[1]));
            for (int i = 2; i<arr_len; i++){
                  if((i-2)>=total_len) break;
                  int row_idx = int(floor(float(i-2)/res[1]));
                  int col_idx =  (i-2) % width;
                  if(col_idx<left_width){
                        res[i] = get_num_from_mat(mat_1,row_idx,col_idx);
                  }else{
                        res[i] = get_num_from_mat(mat_2,row_idx,col_idx-left_width);
                  }
            } 
      }else{
            res[0] = mat_1[0] + mat_2[0];
            res[1] = mat_1[1];
            int total_len = int(round(res[0]*res[1]));
            int height = int(round(res[0]));
            int width = int(round(res[1]));
            int top_height = int(round(mat_1[0]));
            for (int i = 2; i<arr_len ;i++){
                  if((i-2)>=total_len) break;
                  int row_idx = int(floor(float(i-2)/res[1]));
                  int col_idx =  (i-2) % width;
                  if(row_idx<top_height){
                        res[i] = get_num_from_mat(mat_1,row_idx,col_idx);
                  }else{
                        res[i] = get_num_from_mat(mat_2,row_idx-top_height,col_idx);
                  }
            } 
      }
      return res;
}

void as_row_vec(float[arr_len] mat){
    mat[1] = mat[0] * mat[1];
    mat[0] = 1.;
}

void as_col_vec(float[arr_len] mat){
    mat[0] = mat[0] * mat[1];
    mat[1] = 1.;
}

float[arr_len] turn_vec3_into_mat(vec3 arr){
      float[arr_len] res;
      res[0] = 1.;
      res[1] = 3.;
      res[2] = arr[0];
      res[3] = arr[1];
      res[4] = arr[2];
      return res; 
}

float[arr_len] turn_mat3_into_mat(mat3 mat00){
      float[arr_len] res;
      res[0] = 3.;
      res[1] = 3.;
      res[2] = mat00[0][0];
      res[3] = mat00[0][1];
      res[4] = mat00[0][2];
      res[5] = mat00[1][0];
      res[6] = mat00[1][1];
      res[7] = mat00[1][2];
      res[8] = mat00[2][0];
      res[9] = mat00[2][1];
      res[10] = mat00[2][2];
      return res; 
}

mat3 outer_product(vec3 a, vec3 b) {
    return mat3(
        a.x * b.x, a.x * b.y, a.x * b.z,
        a.y * b.x, a.y * b.y, a.y * b.z,
        a.z * b.x, a.z * b.y, a.z * b.z
    );
}


