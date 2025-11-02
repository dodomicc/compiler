#version 330 core


uniform float iTime;
uniform int iFrame;
uniform vec2 iResolution;
uniform sampler2D iChannel3;


out vec4 fragColor;


# define AA 1


#define S smoothstep
float  PI = 3.14159;


// https://iquilezles.org/articles/smin
float smin( float a, float b, float k )
{
    float h = max(k-abs(a-b),0.0);
    return min(a, b) - h*h*0.25/k;
}
// https://iquilezles.org/articles/smin
float smax( float a, float b, float k )
{
    float h = max(k-abs(a-b),0.0);
    return max(a, b) + h*h*0.25/k;
}

vec2 sdSegment( in vec3 p, vec3 a, vec3 b )
{
	vec3 pa = p - a, ba = b - a;
	float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
	return vec2( length( pa - ba*h ), h );
}

float distFromPointToLine(vec3 origin, vec3 dir, vec3 targetPoint){
    vec3 to = targetPoint - origin;
    float h = dot(to,dir)/dot(dir,dir);
    vec3 projPoint = origin + h * dir;
    return length(projPoint - targetPoint);
}



vec3 setCamera(vec3 ro, vec3 ta, vec2 uv) {
    vec3 cw = normalize(ta - ro);
    vec3 cu = normalize(cross(vec3(0.,1.,0.),cw));
    vec3 cv = normalize(cross(cw,cu));
    return normalize(1.6 * cw + cu * uv.x + cv * uv.y);
}

float sdBox( in vec3 p, in vec3 b ) {
    vec3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}


float remap(float val1, float val2, float val3, float val4, float val){
    float h = (val- val1)/(val2 - val1);
    return val3 + h * (val4 - val3);
}



// http://research.microsoft.com/en-us/um/people/hoppe/ravg.pdf
float det( vec2 a, vec2 b ) { return a.x*b.y-b.x*a.y; }
vec3 getClosest( vec2 b0, vec2 b1, vec2 b2 ) 
{
    float a =     det(b0,b2);
    float b = 2.0*det(b1,b0);
    float d = 2.0*det(b2,b1);
    float f = b*d - a*a;
    vec2  d21 = b2-b1;
    vec2  d10 = b1-b0;
    vec2  d20 = b2-b0;
    vec2  gf = 2.0*(b*d21+d*d10+a*d20); gf = vec2(gf.y,-gf.x);
    vec2  pp = -f*gf/dot(gf,gf);
    vec2  d0p = b0-pp;
    float ap = det(d0p,d20);
    float bp = 2.0*det(d10,d0p);
    float t = clamp( (ap+bp)/(2.0*a+b+d), 0.0 ,1.0 );
    return vec3( mix(mix(b0,b1,t), mix(b1,b2,t),t), t );
}

vec2 sdBezier( vec3 a, vec3 b, vec3 c, vec3 p)
{

	vec3 w = normalize( cross( c-b, a-b ) );
	vec3 u = normalize( c-b );
	vec3 v = normalize( cross( w, u ) );

	vec2 a2 = vec2( dot(a-b,u), dot(a-b,v) );
	vec2 b2 = vec2( 0.0 );
	vec2 c2 = vec2( dot(c-b,u), dot(c-b,v) );
	vec3 p3 = vec3( dot(p-b,u), dot(p-b,v), dot(p-b,w) );

	vec3 cp = getClosest( a2-p3.xy, b2-p3.xy, c2-p3.xy );


    
	return vec2( sqrt(dot(cp.xy,cp.xy)+p3.z*p3.z), cp.z );
}


vec3 getBezierPoint(float t, vec3 a, vec3 b, vec3 c){
    float s = 1. - t;
    vec3 res = s * s * a + 2. * s * t * b + t * t * c;
    return res;
}

mat3 getBezierLocalAxis(float t, vec3 a, vec3 b, vec3 c){
    vec3 tangent1 = normalize(b - a);
    vec3 tangent2 = normalize(c - b);
    vec3 localYAxis = normalize(mix(tangent1,tangent2,t));
    vec3 localXAxis = normalize(cross(vec3(b - a),vec3(c - b)));
    vec3 localZAxis = normalize(cross(localXAxis,localYAxis));
    localZAxis *= sign(localZAxis.y);
    return mat3(localXAxis,localYAxis,localZAxis);
}

float sabs(float x, float k){
    return smax(-x,x,k);
}


vec3 setCamera(vec2 uv, vec3 ro, vec3 ta, float FOV){
    vec3 yAxis = normalize(vec3(ta - ro));
    vec3 xAxis = normalize(cross(yAxis,vec3(0.,-1.,0.)));
    vec3 zAxis = normalize(cross(yAxis,xAxis));
    vec3 rd = uv.x * xAxis + uv.y * zAxis + FOV * yAxis;
    return rd;
}




//leaf model
float sdLeaf(vec3 pos, vec3 a, vec3 b, vec3 c, float yMaxLength){

    float d = 1e27;
    vec3 cen = (a+b+ c)/3.;
    

    float d0 = length(pos - cen) - 10.;
    if(d0>10.) return d0 + 10.;

    

    vec2 info = sdBezier(a,b,c,pos);
 
    float t = info.y;
    float s = 1. - t;
    vec3 xDir = normalize(mix(normalize(b-a),normalize(c-b),t));
    vec3 yDir = normalize(cross(b-a,c-b));
    vec3 zDir = normalize(cross(xDir,yDir));
    zDir *= sign(zDir.y);
    vec3 projPoint = s * s * a + 2. * s * t * b + t * t * c;

    float yLength = 0.008 * S(0.8,0.6, t) + yMaxLength * S(0.3, 0.5,t) * S(1.02,0.5,t);
  
 
    
    vec3 projCoord = inverse(mat3(xDir,yDir,zDir)) * (pos - projPoint);
    float deltaZ = 0.1 * S(0.,0.2,projCoord.y);
    float yLengthRatio = sabs(projCoord.y,0.001)/yLength;
    float edgeDelatZ = 0.01 * S (0.,1.,yLengthRatio);
    edgeDelatZ = smax(edgeDelatZ,  0.01 * S (0.8,0.98,t),0.001);
    vec3 projCoord1 = projCoord;
    projCoord1.z -= deltaZ;
    
    float d1 = sdBox(projCoord1,vec3(0.02, yLength,0.01 - edgeDelatZ));
    
    d1 -= 0.001 * sin(-80. * sabs(projCoord.y,0.01) + 120. * t) * S (0.8,0.4,t);
    
    float d2 = info.x - 0.015 * S(0.8,0.4,t);
 

    d = d1;
    d = smin(d,d2,0.001);
    
    return d * 0.1;
    
    
}

float sdPillar(vec3 pos){
    float d = 1e27;
    float height = 10.;
    float h = pos.y/height;
    float theta = atan(pos.x,pos.z);
    d = length(pos.xz) - (1. + 0.2 * S(0.5,1.,abs(sin( 4. * theta))))*mix(1.,0.6,h);
    d = smax(d,h -1.,0.01);
    d = smax(d,-h,0.01);
    h = (pos.y - 10.);
    float xzRatio = 0.8 + 0.2 * sin(4. * PI * h);
    d = smin(d,sdBox(vec3(pos - vec3(0.,10.5,0.)), 0.5 * vec3(1.9* xzRatio,1.,1.9* xzRatio) - 0.1)-0.1, 0.1); 
    return d * 0.7;
}

float sdPillars(vec3 pos){
    vec3 p = pos;
    p.x = 8. - sabs(pos.x,0.1);
    p.z = mod(pos.z,4.) - 2.;
    float d = 1e27;
    d = sdPillar(p);
    d = smax(d,sabs(pos.z,0.1) - 20.,0.1);
    return d * 0.5;
    
}

float sdFloor(vec3 pos){
    vec3 p = pos;
    p.x = mod(p.x + 2.,4.) - 2.;
    p.z = mod(p.z ,4.) - 2.;
    p.y -= -0.5;

    float d1 = sdBox(p,vec3(1.95,0.55,1.95)- 0.1) - 0.1;
    d1 = smax(d1, sabs(pos.x,0.1) - 14., 0.1);
    d1 = smax(d1, sabs(pos.z,0.1) - 24., 0.1);
    
    p = pos;
    p.x = mod(p.x,4.) -2.;
    p.z = mod(p.z + 2.,4.) -2.;
    p.y -= -1.5;
    float d2 = sdBox(p,vec3(1.95,0.5,1.95)- 0.1) - 0.1;
    d2 = smax(d2, sabs(pos.x,0.1) - 16., 0.1);
    d2 = smax(d2, sabs(pos.z,0.1) - 26., 0.1);
    
     
     
     float d = d1;
     d = smin(d,d2,0.1);
     
     
    return d * 0.5;
}

float sdCeil(vec3 pos){
    vec3 p = pos;
    p.y -= 10.6;
    float d = 1e27;
    p.y -= 0.8;
    d = sdBox(p , vec3(10.,0.5,19.5) - 0.1) - 0.1; 
    p.y -= 0.5;
    float d2 = p.y - 5.;
    d2 = smax(d2, p.y - remap(0.,10.,4.,-0.1,sabs(p.x,0.001)),0.01);
    d2 = smax(d2, -(p.y - remap(0.,10.,3.5,-0.6,sabs(p.x,0.001))),0.01);
    d2 = smax(d2,sabs(p.z,0.1) - 19.6,0.001);
    d2 = smax(d2,sabs(p.x,0.1) - 10.8,0.001);
    d = smin(d,d2,0.1);
    return d * 0.5;
}

float sdTemple(vec3 pos){
    //pos.xz *= mat2(vec2(cos(0.6 * iTime),sin(0.6 * iTime)),  vec2(-sin(0.6 * iTime),cos(0.6 * iTime)));
    vec3 p = vec3(pos.z,pos.y,pos.x);
    pos = p;
    float d = smin(sdPillars(pos),sdFloor(pos),0.3);
    d = smin(d, sdCeil(pos),0.01);
    return d;
}

float sdCamera(vec3 pos){
    float d = 1e27;
    d = min(d,length(pos.xy) - 0.5 * pos.z);
    d = max(d,pos.z - 0.3);
    vec3 p = pos;
    p.z -= 0.05;
    float d2 = length(p.xy) - 0.5 * p.z;
    d2 = max(d2,p.z - 0.3);
    d =max(d,-d2); 
    
    return d * 0.1;
}













float type = -1.;
float sdScreen(vec3 pos){
    return sdBox(pos - vec3(0.,3.,-4.  ),vec3(1.5 * 1.7,1.5,0.02)) * 0.5;
    
}




float map(vec3 pos){
    float d = 1e27;
    float screen = sdScreen(pos);
    
    float canteen = sdTemple(3. * (pos - vec3(0.,1.1,4.5))) * 0.2;
    d = screen;

    d = min(d,canteen);
    
    float ground = pos.y - (-.0);
    ground *= 0.6;
    d = min(d,ground ); 
   
    
    if(d==screen){
        type = 1.;
    }else if(d == ground){
        type = 2.;
    }else{
        type = 3.;
    }

    return  d;

}

vec3 calcNor(vec3 pos)
{
    const float eps = 0.001;
    vec3 n = vec3(0.0);
    for( int i=min(iFrame,0); i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(pos+eps*e);
    }
    return normalize(n);
}

float rayMarch(in vec3 ro, in vec3 rd)
{
  float h;
  float t = 0.0;
  vec3 p = ro;	
  for( int j=0; j<1024; j++ )
  {
    h = map(p);
    t+= h;
    p = ro + t*rd; 
    if( h<0.0001 ) return t;
    if( t>200.0 ) return -1.; 
  }

}


float map2(vec3 pos){
    float d = 1e27;


    float canteen = sdTemple(3. * (pos - vec3(0.,1.1,4.5))) * 0.1;



    d = min(d,canteen);
    float ground = pos.y - (-.0);
    ground *= 0.6;
    d = min(d,ground ); 
    if(d==ground){
        type = 2.;
    }else{
        type = 3.;
    }
    


    return  d;

}

vec3 calcNor2(vec3 pos)
{
    const float eps = 0.001;
    vec3 n = vec3(0.0);
    for( int i=min(iFrame,0); i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map2(pos+eps*e);
    }
    return normalize(n);
}

float rayMarch2(in vec3 ro, in vec3 rd)
{
  float h;
  float t = 0.0;
  vec3 p = ro;	
  for( int j=0; j<1024; j++ )
  {
    h = map2(p);
    t+= h;
    p = ro + t*rd; 
    if( h<0.0001 ) return t;
    if( t>100.0 ) return -1.; 
  }

}

vec3 render2(vec2 uv){
    vec3 ro = vec3(0.,3.,-15. + 15. * sin(iTime));
    vec3 ta = vec3(0.,3.,1.);
    
    vec3 col = vec3(0.);
    vec3 rd = setCamera(uv,ro,ta,5.);  
    float dist = rayMarch2(ro,rd);
    if(dist > 0.){
        vec3 pos = ro + dist * rd;
        vec3 nor = calcNor2(pos);
        if(type == 1.){
            vec3 id = abs(floor(5. * pos));
            float blackOrGray = mod(id.x + id.y + id.z,2.)<0.5?1.:0.;
            col = vec3(0.5) + 0.5 * blackOrGray;
        
        }else if(type == 2.){
            vec3 id = abs(floor(0.3 * pos));
            float blackOrGray = mod(id.x + id.y + id.z,2.)<0.5?1.:0.;
            col = blackOrGray == 0. ?vec3(0.5):vec3(0.2,0.2,0.8) ;
        }else{
           col = 0.5 + 0.5 * nor;
        }
        col *= exp(-0.05 * dist);
        //col = vec3(0.5);
    }
    return col;
}




vec3 render(vec2 uv, vec3 ro, vec3 ta){
    vec3 col = vec3(0.);
    vec3 rd = setCamera(uv,ro,ta,1.);  
    float dist = rayMarch(ro,rd);
    if(dist > 0.){
    
        
        vec3 pos = ro + dist * rd;
        vec3 nor = calcNor(pos);
        if(type == 1.){
            col = render2(vec2(pos.x ,pos.y -3.));
            if(abs(nor.z)<0.5 || abs(pos.y - 3.) > 1.45 || abs(pos.x)>1.45 * 1.7){

                vec3 id = abs(floor(1. * pos));
            float blackOrGray = mod(id.x + id.y + id.z,2.)<0.5?1.:0.;
            col = vec3(0.2) + 0.5 * blackOrGray;
            }
       
          
        
        }else if(type == 2.){
            vec3 id = abs(floor(.3 * pos));
            float blackOrGray = mod(id.x + id.y + id.z,2.)<0.5?1.:0.;
            col = blackOrGray == 0. ?vec3(0.5):vec3(0.2,0.2,0.8) ;
        }else{
           col = 0.5 + 0.5 * nor;
        }
        col *= exp(-0.05 * dist);
      
        //col = vec3(0.5);
    }
    return col;
}
void main()
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv ;
    float ft = 0.2 * iTime;




    vec3 ro = vec3(2.5 * cos(0.7 * iTime),10.-3.5 * sin(0.5 * iTime) ,-15.);
    
    
 


    vec3 ta = vec3(0.,-0.3,4.5);

    ta.y = 0.;

    vec3 col = vec3(0.);
    for(int i=-(AA - 1); i<AA; i++){
        for(int j= -(AA - 1); j<AA; j++){
            uv = 1.1 * (gl_FragCoord.xy + 0.5 * vec2(float(i),float(j)) - vec2(0.5,0.5) * iResolution.xy)/iResolution.y;
            col += render(uv,ro,ta);
        }
    }
    col/= pow(float(2. * float(AA) - 1.),2.);
    col = render(uv,ro,ta);

    // Output to screen
    fragColor = vec4(col,1.0);
}
