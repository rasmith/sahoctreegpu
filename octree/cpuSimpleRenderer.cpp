#include "cpuSimpleRenderer.h"

CPUSimpleRenderer::CPUSimpleRenderer(const ConfigLoader& config) : Renderer(config) {
  // Create buffer for ray input 
  rayBuffer.alloc(0, config.bufferType, LOCKED); 
  createRaysOrtho();
  CHK_PRIME(rtpBufferDescCreate(context, 
                                Ray::format, /*RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX*/ 
                                rayBuffer.type(), 
                                rayBuffer.ptr(), 
                                &rayDesc));
  CHK_PRIME(rtpBufferDescSetRange(rayDesc, 0, rayBuffer.count()));

  // Create buffer for returned hit descriptions
  hitBuffer.alloc(rayBuffer.count(), config.bufferType, LOCKED);
  CHK_PRIME(rtpBufferDescCreate(context, 
                                Hit::format, /*RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V*/ 
                                hitBuffer.type(), 
                                hitBuffer.ptr(), 
                                &hitDesc));
  CHK_PRIME(rtpBufferDescSetRange(hitDesc, 0, hitBuffer.count()));
}

void CPUSimpleRenderer::createRaysOrtho(float margin, int yOffset, int yStride) {

  float3& bbmax = scene.bbmax;
  float3& bbmin = scene.bbmin;
  float3 bbspan = bbmax - bbmin;
  
  // set height according to aspect ration of bounding box    
  image.height = (int)(image.width * bbspan.y / bbspan.x);

  float dx = bbspan.x * (1 + 2*margin) / image.width;
  float dy = bbspan.y * (1 + 2*margin) / image.height;
  float x0 = bbmin.x - bbspan.x*margin + dx/2;
  float y0 = bbmin.y - bbspan.y*margin + dy/2;
  float z = bbmin.z - std::max(bbspan.z,1.0f)*.001f;
  int rows = idivCeil( (image.height - yOffset), yStride );
  rayBuffer.alloc( image.width * rows );

  Ray* ray = rayBuffer.ptr();
  float y = y0 + dy*yOffset;
  size_t idx = 0;

  for( int iy=yOffset; iy < image.height; iy += yStride ) {
    float x = x0;
    for( int ix=0; ix < image.width; ix++ ) {
      Ray r = { make_float3(x,y,z), 0, make_float3(0,0,1), 1e34f };
      ray[idx++] = r;
      x += dx;
    }
    y += dy*yStride;
  }  
}

void CPUSimpleRenderer::render() {
  RTPquery query;
  CHK_PRIME(rtpQueryCreate(model, RTP_QUERY_TYPE_CLOSEST, &query));
  CHK_PRIME(rtpQuerySetRays(query, rayDesc));
  CHK_PRIME(rtpQuerySetHits(query, hitDesc));
  CHK_PRIME(rtpQueryExecute(query, 0 /* hints */));
}

