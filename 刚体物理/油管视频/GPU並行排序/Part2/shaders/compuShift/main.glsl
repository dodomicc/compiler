// Created by sebastien durand - 2015
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
//-----------------------------------------------------

// Change this to improve quality - Rq only applied on edge
#define ANTIALIASING 5

// decomment this to see where antialiasing is applied
//#define SHOW_EDGES

#define RAY_STEP 48
//#define NOISE_SKIN
#define ZERO min(0,iFrame)
#define PI 3.14159279







float closest;			// min distance to chameleon on the ray (use for glow light) 

// ----------------------------------------------------

float hash( float n ) { return fract(sin(n)*43758.5453123); }



// ----------------------------------------------------

bool intersectSphere(in vec3 ro, in vec3 rd, in vec3 c, in float r) {
    ro -= c;
	float b = dot(rd,ro), d = b*b - dot(ro,ro) + r*r;
	return (d>0. && -sqrt(d)-b > 0.);
}

// ----------------------------------------------------

float udRoundBox( vec3 p, vec3 b, float r ){
  	return length(max(abs(p)-b,0.))-r;
}

// capsule with bump in the middle -> use for arms and legs
vec2 sdCapsule( vec3 p, vec3 a, vec3 b, float r ) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa,ba)/dot(ba,ba), 0., 1. );
    float dd = cos(3.14*h*2.5);  // Little adaptation
    return vec2(length(pa - ba*h) - r*(1.-.1*dd+.4*h), 30.-15.*dd); 
}

vec2 smin(in vec2 a, in vec2 b, in float k ) {
	float h = clamp( .5 + (b.x-a.x)/k, 0., 1. );
	return mix( b, a, h ) - k*h*(1.-h);
}

float smin(in float a, in float b, in float k ) {
	float h = clamp( .5 + (b-a)/k, 0., 1. );
	return mix(b, a, h) - k*h*(1.-h);
}

vec2 min2(in vec2 a, in vec2 b) {
	return a.x<b.x?a:b;
}

// ----------------------------------------------------


    
vec2 support(vec3 p, vec2 c, float th) {

    float d1 = length(p-vec3(0,-6.5,0)) - 3.;          
    float d = length(max(abs(p-vec3(0,-2,.75))-vec3(.5,2.5,.1),0.))-.11;     

    d = min(d, max(length(max(abs(p)-vec3(4,3,.1),0.))-.1,
                  -length(max(abs(p)-vec3(3.5,2.5,.5),0.))+.1));
    return min2(vec2(d1,-105.),
        min2(vec2(d,-100.), 
        vec2(length(max(abs(p-vec3(0,0,.2))-vec3(3.4,2.4,.01),0.))-.3, -103.)));
}


//----------------------------------------------------------------------

vec2 map(in vec3 pos) {
    // Ground
    vec2 res1 = vec2( pos.y+4.2, -101.0 );
    // Screen
	res1 = min2(support(pos, vec2(.1,15.), 0.05), res1);
    
   
        return res1;
    
}


//----------------------------------------------------------------------
#define EDGE_WIDTH 0.15

vec2 castRay(in vec3 ro, in vec3 rd, in float maxd, inout float hmin) {
    closest = 9999.; // reset closest trap
	float precis = .0006, h = EDGE_WIDTH+precis, t = 2., m = -1.;
    hmin = 0.;
    for( int i=ZERO; i<RAY_STEP; i++) {
        if( abs(h)<t*precis || t>maxd ) break;
        t += h;
	    vec2 res = map(ro+rd*t);
        if (h < EDGE_WIDTH && res.x > h + 0.001) {
			hmin = 10.0;
		}
        h = res.x;
	    m = res.y;
    }
	//if (hmin != h) hmin = 10.;
    if( t>maxd ) m = -200.0;
    return vec2( t, m );
}

float softshadow( in vec3 ro, in vec3 rd, in float mint, in float maxt, in float k) {
	float res = 1.0;
    float t = mint;
    for( int i=ZERO; i<26; i++ ) {
		if( t>maxt ) break;
        float h = map( ro + rd*t ).x;
        res = min( res, k*h/t );
        t += h;
    }
    return clamp( res, 0., 1.);
}

// normal with kali edge finder
float Edge=0.;
vec3 calcNormal(vec3 p, vec3 rd, float t) { 
    float pitch = .2 * t / iResolution.x; 
	pitch = 0.01;

	vec3 e = vec3(0.0,2.*pitch,0.0);
	float d1=map(p-e.yxx).x,d2=map(p+e.yxx).x;
	float d3=map(p-e.xyx).x,d4=map(p+e.xyx).x;
	float d5=map(p-e.xxy).x,d6=map(p+e.xxy).x;
	float d=map(p).x;
    
	Edge=abs(d-0.5*(d2+d1))+abs(d-0.5*(d4+d3))+abs(d-0.5*(d6+d5)); //edge finder
	Edge=min(1.,pow(Edge,.55)*15.);
    
    vec3 grad = vec3(d2-d1,d4-d3,d6-d5);
	return normalize(grad);
}




float calcAO( in vec3 pos, in vec3 nor) {
	float totao = 0.0;
    float sca = 1.0;
    for( int aoi=ZERO; aoi<5; aoi++ ) {
        float hr = 0.01 + 0.05*float(aoi);
        vec3 aopos =  nor * hr + pos;
        float dd = map( aopos ).x;
        totao += (hr - dd)*sca;
        sca *= .75;
    }
    return clamp( 1.0 - 4.0*totao, 0.0, 1.0 );
}

vec3 screen(in vec2 uv, vec3 scrCol) {

	vec3 col =scrCol;  
    return col*col;
}

// clean lines on the ground
float isGridLine(vec2 p, vec2 v) {
    vec2 k = smoothstep(.1,.9,abs(mod(p+v*.5, v)-v*.5)/.08);
    return k.x * k.y;
}

vec3 render( in vec3 ro, in vec3 rd, inout float hmin) { 
    // Test bounding sphere (optim)
    vec2 res = castRay(ro,rd,60.0, hmin);
    float distCham = abs(closest);
#ifdef SHOW_EDGES
     if( res.y>-150.)  {
           vec3 pos = ro + res.x*rd;
     	vec3 nor = calcNormal(pos, rd, res.x);
     }
    return vec3(1);
#else
    
    float t = res.x;
	float m = res.y;
    vec3 cscreen = vec3(sin(.1+1.1*iTime), cos(.1+1.1*iTime),.5);
    cscreen *= cscreen;
 
    vec3 col = vec3(0.);
	float dt;

  //floor(.01+10.5*iTime)));
    
    if( m>-150.)  { 
        vec3 pos = ro + t*rd;
        vec3 nor = calcNormal(pos, rd, t);

        if (m<-104.5) {  // bottom of screen
            col = vec3(.92);
             
        } else if (m<-102.5) {
           	if (pos.z<0.) { // screen
            	col = vec3(0.5);
                vec2 localPos = pos.xy ;
                
                vec2 localResolution = 2. * vec2(3.5,2.5)  ;
                localPos += 0.5 * localResolution;
                vec2 localUV = localPos/localResolution;
                col = texture(iChannel0,localUV).xyz;
                
               
            } else { // back of screen
                col = vec3(.92);
            	
            }
        } else if (m<-101.5) {
            col = .2+3.5*cscreen;
            
        } else if(m<-100.5) {  // Ground
            float f = mod( floor(2.*pos.z) + floor(2.*pos.x), 2.0);
            col = 0.4 + 0.1*f*vec3(1.);
            col = .1+.9*col*isGridLine(pos.xz, vec2(2.));
            dt = dot(normalize(pos-vec3(-4,-4,0)), vec3(0,0,-1));
            col += (dt>0.) ? (.75+.3)*dt*cscreen: vec3(0);     
    		//col = clamp(col,0.,1.);
        } else {  // Screen
            col = vec3(.92);
            
        }
		
        float ao = calcAO( pos, nor );

		vec3 lig = normalize( vec3(-0.6, 0.7, -0.5) );
		float amb = clamp( 0.5+0.5*nor.y, 0.0, 1.0 );
        float dif = clamp( dot( nor, lig ), 0.0, 1.0 );
        float bac = clamp( dot( nor, normalize(vec3(-lig.x,0.0,-lig.z))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);

		float sh = 1.0;
		if( dif>0.02 ) { 
            sh = softshadow( pos, lig, 0.02, 13., 8.0 ); 
            dif *= sh; 
        }

		vec3 brdf = vec3(0.0);
		brdf += 1.80*amb*vec3(0.10,0.11,0.13)*ao;
        brdf += 1.80*bac*vec3(0.15,0.15,0.15)*ao;
        brdf += 0.8*dif*vec3(1.00,0.90,0.70)*ao;

		float pp = clamp( dot( reflect(rd,nor), lig ), 0.0, 1.0 );
		float spe = 1.2*sh*pow(pp,16.0);
		float fre = ao*pow( clamp(1.0+dot(nor,rd),0.0,1.0), 2.0 );

		col = col*brdf*(.5+.5*sh) + vec3(.25)*col*spe + 0.2*fre*(0.5+0.5*col);
        
        float rimMatch =  1. - max( 0. , dot( nor , -rd ) );
        col += vec3((.1+cscreen*.1 )* pow( rimMatch, 10.));
	}

	col *= 2.5*exp( -0.1*t );
    float BloomFalloff = 15000.; //mix(1000.,5000., Anim);
    col += .5*cscreen/(1.+distCham*distCham*distCham*BloomFalloff);
    
	return vec3( clamp(col,0.0,1.0) );
#endif    
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    // animation
    float GlobalTime = iTime; // + .1*hash(iTime);
    
    
    float time = iTime;
    float start_time  = 0.;
    float end_time = decode_float(texture(iChannel1,vec2(0.,0.)));
    float alpha = smoothstep(0.,end_time,iTime);
    

    
    // camera	
    float start = -20.;
    float end = -10.;
    
    vec3 ro = vec3( 0.,0.,mix(start,end,smoothstep(0.,1.,alpha)));
    vec3 ta = vec3( 0., 0., 0. );

    // camera tx
    vec3 cw = normalize( ta-ro );
    vec3 cp = vec3( 0.0, 1.0, 0.0 );
    vec3 cu = normalize( cross(cw,cp) );
    vec3 cv = normalize( cross(cu,cw) );

    // render
    vec3 colorSum = vec3(0);
    int nbSample = 0;
    
 #if (ANTIALIASING == 1)	
	int i=0;
#else
	for (int i=ZERO;i<ANTIALIASING;i++) {
#endif
        float randPix = 0.; //hash(iTime); // Use frame rate to improve antialiasing ... not sure of result
		vec2 subPix = .4*vec2(cos(randPix+6.28*float(i)/float(ANTIALIASING)),
                              sin(randPix+6.28*float(i)/float(ANTIALIASING)));
		//vec3 ray = Ray(2.0,fragCoord.xy+subPix);
		vec2 q = (fragCoord.xy+subPix)/iResolution.xy;
		//vec2 q = (fragCoord.xy+.4*vec2(cos(6.28*float(i)/float(ANTIALIASING)),sin(6.28*float(i)/float(ANTIALIASING))))/iResolution.xy;
        vec2 p = -1.0+2.0*q;
        p.x *= iResolution.x/iResolution.y;
        vec3 rd = normalize( p.x*cu + p.y*cv + 2.5*cw );
        
        nbSample++;
        float hmin = 100.;
        colorSum += sqrt(render( ro, rd, hmin));
        
#ifdef SHOW_EDGES
 		colorSum = vec3(1);
        if (Edge>0.3) colorSum = vec3(.6);  
        if (hmin>0.5) colorSum = vec3(0,0,0);   
        break;
#endif
        
#if (ANTIALIASING > 1)
        // optim : use antialiasing only on objects edges //exit if far from objects
        if (Edge<0.3 && hmin<0.5 ) break;
	}
#endif
    
    fragColor = vec4(colorSum/float(nbSample), 1.);
}