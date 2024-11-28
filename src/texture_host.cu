
#include <comm_mpi.h>
#include <constants.h>
#include <texture.h>
#include <texture_host.h>

namespace CULQCD {

#define BINDTEXNAMEID(name, array, id)                                         \
  CUDA_SAFE_CALL(cudaBindTexture(0, name##_a##id, array##.array##id))

// TODO: before bind to texture, check the number of elements and then bind if
// inside limit 		also if number of elements is not supported...
// set ->
// UseTextureMemory(false);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
static void *gauge_single_ptr = NULL;
static void *gauge_double_ptr = NULL;

static void *delta_single_ptr = NULL;
static void *delta_double_ptr = NULL;

static void *gx_single_ptr = NULL;
static void *gx_double_ptr = NULL;

static void *ilambda_single_ptr = NULL;
static void *ilambda_double_ptr = NULL;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GAUGE_TEXTURE(complexs *pgauge, bool bind) {
  if (PARAMS::UseTex && bind) {
    BIND_GAUGE_TEXTURE(pgauge);
    return;
  }
  if (!bind) {
    gauge_single_ptr = NULL;
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_gauge_float));
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("UNBind Gauge Texture in SINGLE Precision\n");
  }
}
void GAUGE_TEXTURE(complexd *pgauge, bool bind) {
  if (PARAMS::UseTex && bind) {
    BIND_GAUGE_TEXTURE(pgauge);
    return;
  }
  if (!bind) {
    gauge_double_ptr = NULL;
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_gauge_double));
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("UNBind Gauge Texture in DOUBLE Precision\n");
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void BIND_GAUGE_TEXTURE(complexs *pgauge) {
  if (gauge_single_ptr == (void *)pgauge) {
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("Gauge array already binded to Texture\n");
  } else {
    gauge_single_ptr = (void *)pgauge;
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_gauge_float, pgauge));
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("Bind Gauge array Texture in SINGLE Precision\n");
  }
}
void BIND_GAUGE_TEXTURE(complexd *pgauge) {
  if (gauge_double_ptr == (void *)pgauge) {
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("Gauge array already binded to Texture\n");
  } else {
    gauge_double_ptr = (void *)pgauge;
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_gauge_double, pgauge));
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("Bind Gauge array to Texture in DOUBLE Precision\n");
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DELTA_TEXTURE(complexs *delta, bool bind) {
  if (PARAMS::UseTex && bind) {
    BIND_DELTA_TEXTURE(delta);
    return;
  }
  if (!bind) {
    delta_single_ptr = NULL;
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_delta_float));
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("UNBind Delta Texture in SINGLE Precision\n");
  }
}
void DELTA_TEXTURE(complexd *delta, bool bind) {
  if (PARAMS::UseTex && bind) {
    BIND_DELTA_TEXTURE(delta);
    return;
  }
  if (!bind) {
    delta_double_ptr = NULL;
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_delta_double));
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("UNBind Delta Texture in DOUBLE Precision\n");
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void BIND_DELTA_TEXTURE(complexs *delta) {
  if (delta_single_ptr == (void *)delta) {
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("Delta array already binded to Texture\n");
  } else {
    delta_single_ptr = (void *)delta;
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_delta_float, delta));
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("Bind Delta array Texture in SINGLE Precision\n");
  }
}
void BIND_DELTA_TEXTURE(complexd *delta) {
  if (delta_double_ptr == (void *)delta) {
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("Delta array already binded to Texture\n");
  } else {
    delta_double_ptr = (void *)delta;
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_delta_double, delta));
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("Bind Delta array to Texture in DOUBLE Precision\n");
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GX_TEXTURE(complexs *gx, bool bind) {
  if (PARAMS::UseTex && bind) {
    BIND_GX_TEXTURE(gx);
    return;
  }
  if (!bind) {
    gx_single_ptr = NULL;
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_gx_float));
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("UNBind GX Texture in SINGLE Precision\n");
  }
}
void GX_TEXTURE(complexd *gx, bool bind) {
  if (PARAMS::UseTex && bind) {
    BIND_GX_TEXTURE(gx);
    return;
  }
  if (!bind) {
    gx_double_ptr = NULL;
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_gx_double));
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("UNBind GX Texture in DOUBLE Precision\n");
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void BIND_GX_TEXTURE(complexs *gx) {
  if (gx_single_ptr == (void *)gx) {
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("GX array already binded to Texture\n");
  } else {
    gx_single_ptr = (void *)gx;
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_gx_float, gx));
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("Bind GX array Texture in SINGLE Precision\n");
  }
}
void BIND_GX_TEXTURE(complexd *gx) {
  if (gx_double_ptr == (void *)gx) {
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("GX array already binded to Texture\n");
  } else {
    gx_double_ptr = (void *)gx;
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_gx_double, gx));
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("Bind GX array to Texture in DOUBLE Precision\n");
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ILAMBDA_TEXTURE(complexs *ilambda, bool bind) {
  if (PARAMS::UseTex && bind) {
    BIND_ILAMBDA_TEXTURE(ilambda);
    return;
  }
  if (!bind) {
    ilambda_single_ptr = NULL;
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_lambda_float));
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("UNBind ILAMBDA Texture in SINGLE Precision\n");
  }
}
void ILAMBDA_TEXTURE(complexd *ilambda, bool bind) {
  if (PARAMS::UseTex && bind) {
    BIND_ILAMBDA_TEXTURE(ilambda);
    return;
  }
  if (!bind) {
    ilambda_double_ptr = NULL;
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_lambda_double));
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("UNBind ILAMBDA Texture in DOUBLE Precision\n");
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void BIND_ILAMBDA_TEXTURE(complexs *ilambda) {
  if (ilambda_single_ptr == (void *)ilambda) {
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("ILAMBDA array already binded to Texture\n");
  } else {
    ilambda_single_ptr = (void *)ilambda;
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_lambda_float, ilambda));
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("Bind ILAMBDA array Texture in SINGLE Precision\n");
  }
}
void BIND_ILAMBDA_TEXTURE(complexd *ilambda) {
  if (ilambda_double_ptr == (void *)ilambda) {
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("ILAMBDA array already binded to Texture\n");
  } else {
    ilambda_double_ptr = (void *)ilambda;
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_lambda_double, ilambda));
    if (getVerbosity() == DEBUG_VERBOSE)
      printfCULQCD("Bind ILAMBDA array to Texture in DOUBLE Precision\n");
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void UNBIND_ALL_TEXTURES() {
  if (getVerbosity() == DEBUG_VERBOSE)
    printfCULQCD("UNBind ALL Textures in SINGLE Precision\n");
  if (!gauge_single_ptr) {
    gauge_single_ptr = NULL;
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_gauge_float));
  }
  if (!delta_single_ptr) {
    delta_single_ptr = NULL;
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_delta_float));
  }
  if (!gx_single_ptr) {
    gx_single_ptr = NULL;
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_gx_float));
  }
  if (!ilambda_single_ptr) {
    ilambda_single_ptr = NULL;
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_lambda_float));
  }
  if (getVerbosity() == DEBUG_VERBOSE)
    printfCULQCD("UNBind ALL Textures in DOUBLE Precision\n");
  if (!gauge_double_ptr) {
    gauge_double_ptr = NULL;
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_gauge_double));
  }
  if (!delta_double_ptr) {
    delta_double_ptr = NULL;
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_delta_double));
  }
  if (!delta_double_ptr) {
    gx_double_ptr = NULL;
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_gx_double));
  }
  if (!ilambda_double_ptr) {
    ilambda_double_ptr = NULL;
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_lambda_double));
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace CULQCD