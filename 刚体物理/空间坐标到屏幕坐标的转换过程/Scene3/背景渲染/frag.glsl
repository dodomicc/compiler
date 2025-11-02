#version 330 core


uniform float iTime;
uniform int iFrame;
uniform vec2 iResolution;
uniform sampler2D iChannel3;


out vec4 fragColor;


#define S smoothstep
# define AA 2
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

float sdArrow(vec3 pos, vec3 point1, vec3 point2, float r){
    float a = dot(pos-point1,point2-point1);
    float b = dot(point1-point2,point1-point2);
    float h = clamp(a/b,0.,1.);
    float maxR = 2. * r;

    float main = length(pos - (point1 + h * (point2- point1))) -r;
    
    main = max(main, -a/b);
    vec3 conePoint1 = point2 - r * normalize(point2 - point1);
    vec3 conePoint2 = conePoint1 + 3. * r * normalize(point2 - point1);
    float coneH =  clamp(dot(pos-conePoint1,conePoint2-conePoint1)/dot(conePoint2-conePoint1,conePoint2-conePoint1),0.,1.);
    float cone =  length(pos - ( conePoint1+ coneH * (conePoint2- conePoint1))) - mix(maxR,0.,coneH );
    cone = max(cone, -dot(pos-conePoint1,conePoint2-conePoint1)/dot(conePoint2-conePoint1,conePoint2-conePoint1));
    
    return min(main,cone);
}


vec2 projectionPoint(vec3 pos, vec3 ro, vec3 ta){
    vec3 yAxis = normalize(vec3(ta - ro));
    vec3 xAxis = normalize(cross(yAxis,vec3(0.,-1.,0.)));
    vec3 zAxis = normalize(cross(yAxis,xAxis));
    mat3 mm = mat3(xAxis,yAxis,zAxis);
    vec3 coord = inverse(mm) * (pos - ro);
    float t = coord.y;
    float x = coord.x/t;
    float y = coord.z/t;
    return vec2(x,y);
    
}









float type = -1.;


float screenT = 1.;
float groundT = 2.;
float cameraOriginT = 3.;
float cameraLookAtT = 4.;
float randomPointT = 5.;
float elseT = 6.;




float screenZ = 7.;
float screenY = 5.;
float screenHalfHeight = 2.;

vec3 cameraOriginPoint ;
vec3 cameraLookAtPoint ;
vec3 randomPointCoord;




//P0，Q，y,x,z,P,t,(x',y')


float sdSphere(vec3 pos, vec3 cen, float r){
    return length(pos - cen) - r;
}

float sdScreen(vec3 pos){
    return sdBox(pos - vec3(0.,screenY,screenZ),vec3(screenHalfHeight * 1.7,screenHalfHeight,0.01)) * 0.5;
    
}

float computeFOV(vec3 ro, vec3 ta){
    float FOV = screenZ - ro.z;
    return FOV;
}



float map(vec3 pos){
    float d = 1e27;
    float screen = 1e27, 
    ground = 1e27, 
    cameraOrigin = 1e27, 
    cameraLookAt = 1e27, 
    randomPoint = 1e27,
    originLookAtLine = 1e27,
    xArrow = 1e27,
    yArrow = 1e27,
    zArrow = 1e27,
    frustumEdge = 1e27;
    
    
    screen = sdScreen(pos);
    ground = pos.y * 0.6;

    

    d = screen;
    d = min(d,ground);
    


   
        vec3 yDir = vec3(0.,0.,1.);
        vec3 xDir = cross(vec3(0.,0.,1.),vec3(0.,1.,0.));
        vec3 zDir = cross(xDir,yDir);
        yArrow = sdArrow(pos,cameraOriginPoint,cameraOriginPoint + 2. * yDir ,0.05);
        xArrow = sdArrow(pos,cameraOriginPoint,cameraOriginPoint + 2. * xDir ,0.05);
        zArrow = sdArrow(pos,cameraOriginPoint,cameraOriginPoint + 2. * zDir ,0.05);
        
      
        cameraOrigin = sdSphere(pos,cameraOriginPoint,0.2 );
        vec3 p = pos;
        p.x = abs(p.x);
        p.y = abs(pos.y - screenY);
        vec3 start = vec3(0.,0.,cameraOriginPoint.z);
        vec3 dir = vec3(1.7 * screenHalfHeight,screenHalfHeight,screenZ) - start;
        vec2 frustumEdgeInfo = sdSegment(p,start,start + 1.2 * dir);
        frustumEdge = 20.;

        frustumEdge = max(frustumEdge,-0.5 + S(-0.5,0.5, sin(100. * frustumEdgeInfo.y)));
        frustumEdge *= 0.3;
        randomPoint = sdSphere(pos,randomPointCoord,0.6 ) * 0.5 ;
        
         vec2 d2Info = sdSegment(pos,cameraOriginPoint,randomPointCoord);
        float d2 = d2Info.x - 0.015;
  
        d2 = max(d2,-0.5 + S(-0.5,0.5, sin(100. * d2Info.y)));
        frustumEdge = min(frustumEdge,d2 * 0.5);
       
         
    
    
    d = min(d,cameraOrigin);
    d = min(d,cameraLookAt);
    d = min(d,originLookAtLine);
    d = min(d,xArrow);
    d = min(d,yArrow);
    d = min(d,zArrow);
    d = min(d,frustumEdge);
    d = min(d,randomPoint);
   
    
    if(d==screen){
        type = screenT;
    }else if(d == ground){
        type = groundT;
    }else{
        type = elseT;
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
//屏幕上画出投影点并打出x'和y'
vec3 renderScreen(vec2 screenUV){
    vec3 col = vec3(0.);


        vec3 ro = vec3(screenUV.x,screenUV.y + screenY,screenZ);
        vec3 rd = (screenZ - cameraOriginPoint.z)*vec3(0.,0.,1.) + vec3(screenUV,0.);
        rd = normalize(rd);
        ro += 0.05 * rd;
        float dist = rayMarch(ro,rd);
        if(dist > 0.){
            vec3 pos = ro + dist * rd;
            vec3 nor = calcNor(pos);
            if(type == groundT){
                vec3 id = abs(floor(.3 * pos));
                float blackOrGray = mod(id.x + id.y + id.z,2.)<0.5?1.:0.;
                col = blackOrGray == 0. ?vec3(0.5):vec3(0.2,0.2,0.9) ;
            }else{
               col = 0.5 + 0.5 * nor;
            }
            col *= exp(-0.05 * dist);
            
        }
        
        float ratio = remap(cameraOriginPoint.z,randomPointCoord.z,0.,1.,screenZ); 
        vec3 intersect = mix(cameraOriginPoint,randomPointCoord,ratio);
        intersect.y -= screenY;
        if(sdBox(vec3(screenUV,0.) - vec3(intersect.xy,0.), vec3(1.8 * 0.7,0.8 * 0.7,0.0001))<0.){
        //导入(x',y')的贴图
            float x = remap(intersect.x - 1.8 * 0.7, intersect.x + 1.8 * 0.7,0.,1./5 - 0.03,screenUV.x);
            float y = remap(intersect.y - 0.8 * 0.7 , intersect.y + 0.8 * 0.7,7./8.,1.,screenUV.y);
            col = mix(col,vec3(1.),step(0.5, texture(iChannel3,vec2(x,y)).x)); 
        }
        
        
        
    
    return col * 1.6;
}


vec3 render(vec2 uv, vec3 ro, vec3 ta){
    vec3 col = vec3(0.);
    vec3 rd = setCamera(uv,ro,ta,1.);  
    float dist = rayMarch(ro,rd);
    if(dist > 0.){
        vec3 pos = ro + dist * rd;
        vec3 nor = calcNor(pos);
        if(type == screenT){
            col = renderScreen(vec2(pos.x,pos.y - screenY));
            if(abs(nor.z)<0.5 || abs(pos.y - screenY) > screenHalfHeight * 0.95 || abs(pos.x)>screenHalfHeight * 0.95 * 1.7){
                vec3 id = abs(floor(1. * pos));
                float blackOrGray = mod(id.x + id.y + id.z,2.)<0.5?1.:0.;
                col = vec3(0.5) + 0.5 * blackOrGray;
            }
        }else if(type == groundT){
            vec3 id = abs(floor(.3 * pos));
            float blackOrGray = mod(id.x + id.y + id.z,2.)<0.5?1.:0.;
            col = blackOrGray == 0. ?vec3(0.5):vec3(0.2,0.2,0.9) ;
        }else{
           col = 0.5 + 0.5 * nor;
        }
        col *= exp(-0.05 * dist);
      
        //col = vec3(0.5);
    }
    return col * 1.6;
}

//标出相机的原点P0 -- sum1

//标出相机的look At Q点并连接 --sum2

//标出y轴 --sum3

//标出x轴 --sum4

//标出z轴以及生成相机覆盖线 --sum5

//标出任意点P并虚线连接，--sum6

//并且标出步长t --sum7
vec3 textsampler(vec2 uv, vec2 cen, vec2 xInt, vec2 yInt){
    vec3 col = vec3(0.);
    float temp = abs(uv - cen).x;
    
    if(max(abs(uv - cen).x,abs(uv - cen).y)>0.03) return col;
    float x = remap(cen.x - 0.03,cen.x + 0.03,xInt.x,xInt.y,uv.x);
    float y = remap(cen.y - 0.03,cen.y + 0.03,yInt.x,yInt.y,uv.y);
    col = texture(iChannel3,vec2(x,y)).xyz;
    return col;
}

vec3 renderText(vec2 uv, vec3 ro, vec3 ta){
    vec3 col = vec3(0.);



    vec2 camerOriginProjPoint = projectionPoint(cameraOriginPoint,ro,ta);
    //导入P0点贴图
    col = mix(col,vec3(1.),textsampler(uv,camerOriginProjPoint,vec2(0.,1./9.),vec2(0.,1./8.)).x);
    vec2 randomPointProjPoint = projectionPoint(randomPointCoord,ro,ta);
        //导入P点贴图
    col = mix(col,vec3(1.),textsampler(uv,randomPointProjPoint,vec2(0.,1./9.),vec2(5./8.,6./8.)).x);
        
    vec2 tPointProjPoint = projectionPoint((cameraOriginPoint + randomPointCoord)/2.,ro,ta);
        //导入t点贴图
    col = mix(col,vec3(1.),textsampler(uv,tPointProjPoint,vec2(0.,1./9.),vec2(6./8.,7./8.)).x);

    vec3 yDir = vec3(0.,0.,1);
    vec3 xDir = cross(vec3(0.,0.,1.),vec3(0.,1.,0.));
    vec3 zDir = cross(xDir,yDir);
        
    vec2 yAxisProjPoint = projectionPoint(cameraOriginPoint + 2. * yDir,ro,ta);

    //导入y方向贴图
    col = mix(col,vec3(1.),textsampler(uv,yAxisProjPoint,vec2(0.,1./9.),vec2(2./8.,3./8.)).x);
      
    vec2 xAxisProjPoint = projectionPoint(cameraOriginPoint + 2.* xDir,ro,ta);
    //导入x方向贴图
    col = mix(col,vec3(1.),textsampler(uv,xAxisProjPoint,vec2(0.,1./9.),vec2(3./8.,4./8.)).x);
        
    vec2 zAxisProjPoint = projectionPoint(cameraOriginPoint + 2.* zDir,ro,ta);
        //导入z方向贴图
    col = mix(col,vec3(1.),textsampler(uv,zAxisProjPoint,vec2(0.,1./9.),vec2(4./8.,5./8.)).x);
    


    return col;
}
void main(){

    cameraOriginPoint  = vec3(0.,screenY,1.);
    cameraLookAtPoint  = vec3(0.,screenY,10.);
    randomPointCoord = vec3(3.2 * sin(0.4 * iTime), screenY + 2.4 * sin(0.6 * iTime), screenZ +  + 4. + 3. * sin(0.5 * iTime));
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv =  (gl_FragCoord.xy - vec2(0.25,0.5) * iResolution.xy)/iResolution.y;
    float ft = 0.2 * iTime;




    vec3 ro = vec3(8. + 1.5 * cos(0.7 * iTime),17.-3.5 * sin(0.5 * iTime) ,1.5 +1.5 * sin(iTime));
    
    
 


    vec3 ta = vec3(0.,0.,screenZ - 3.);



    vec3 col = vec3(0.);
    for(int i=-(AA - 1); i<AA; i++){
        for(int j= -(AA - 1); j<AA; j++){
            uv = 1.1 * (gl_FragCoord.xy + 0.5 * vec2(float(i),float(j)) - vec2(0.18,0.5) * iResolution.xy)/iResolution.y;
            col += render(uv,ro,ta);
        }
    }
    col/= pow(float(2. * float(AA) - 1.),2.);
    vec3 col2 = renderText(uv,ro,ta);
    col = mix(col,col2,S(0.5,0.9,col2.x));
    


    // Output to screen
    fragColor = vec4(col,1.0);
}