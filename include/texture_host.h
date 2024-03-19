
#ifndef TEXTURE_HOST_H
#define TEXTURE_HOST_H

#include <complex.h>

namespace CULQCD{
////////////////////////////////////////////////////////////
void GAUGE_TEXTURE(complexs *pgauge, bool bind);//binds if bind==true and PARAMS::UseTex==true
void GAUGE_TEXTURE(complexd *pgauge, bool bind);//unbind if bind==false 
void BIND_GAUGE_TEXTURE(complexs *pgauge);//bind to texture if this array was not binded yet
void BIND_GAUGE_TEXTURE(complexd *pgauge);
////////////////////////////////////////////////////////////
void DELTA_TEXTURE(complexs *delta, bool bind);//binds if bind==true and PARAMS::UseTex==true
void DELTA_TEXTURE(complexd *delta, bool bind);//unbind if bind==false 
void BIND_DELTA_TEXTURE(complexs *delta);
void BIND_DELTA_TEXTURE(complexd *delta);
////////////////////////////////////////////////////////////
void GX_TEXTURE(complexs *gx, bool bind);//binds if bind==true and PARAMS::UseTex==true
void GX_TEXTURE(complexd *gx, bool bind);//unbind if bind==false 
void BIND_GX_TEXTURE(complexs *gx);//bind to texture if this array was not binded yet
void BIND_GX_TEXTURE(complexd *gx);
////////////////////////////////////////////////////////////
void ILAMBDA_TEXTURE(complexs *ilambda, bool bind);//binds if bind==true and PARAMS::UseTex==true
void ILAMBDA_TEXTURE(complexd *ilambda, bool bind);//unbind if bind==false 
void BIND_ILAMBDA_TEXTURE(complexs *ilambda);//bind to texture if this array was not binded yet
void BIND_ILAMBDA_TEXTURE(complexd *ilambda);
////////////////////////////////////////////////////////////
void UNBIND_ALL_TEXTURES();
////////////////////////////////////////////////////////////
}
#endif 
