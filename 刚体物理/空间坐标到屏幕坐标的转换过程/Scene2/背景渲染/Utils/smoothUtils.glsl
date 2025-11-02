float smin( float a, float b, float k )
{
    k *= 4.0;
    float h = max( k-abs(a-b), 0.0 )/k;
    return min(a,b) - h*h*k*(1.0/4.0);
}



float smax(float a, float b, float k){
    return -smin(-a,-b,k);
}

float sabs(float x, float k){
    return smax(-x,x,k);
}