// version 330
float remap(float val1,float val2, float val3, float val4, float x){
    float h = (x-val1)/(val2-val1);
    return val3 + h * (val4 - val3);
}


