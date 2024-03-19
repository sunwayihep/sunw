  #define LAUNCH_KERNEL(kernel, tp, stream, arg, ...)     \
  switch (tp.block.x) {             \
  case 32:                \
    kernel<32,__VA_ARGS__>            \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 64:                \
    kernel<64,__VA_ARGS__>            \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 96:                \
    kernel<96,__VA_ARGS__>            \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 128:               \
    kernel<128,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 160:               \
    kernel<160,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 192:               \
    kernel<192,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 224:               \
    kernel<224,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 256:               \
    kernel<256,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 288:               \
    kernel<288,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 320:               \
    kernel<320,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 352:               \
    kernel<352,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 384:               \
    kernel<384,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 416:               \
    kernel<416,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 448:               \
    kernel<448,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 480:               \
    kernel<480,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 512:               \
    kernel<512,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 544:               \
    kernel<544,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 576:               \
    kernel<576,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 608:               \
    kernel<608,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 640:               \
    kernel<640,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 672:               \
    kernel<672,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 704:               \
    kernel<704,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 736:               \
    kernel<736,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 768:               \
    kernel<768,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 800:               \
    kernel<800,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 832:               \
    kernel<832,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 864:               \
    kernel<864,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 896:               \
    kernel<896,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 928:               \
    kernel<928,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 960:               \
    kernel<960,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 992:               \
    kernel<992,__VA_ARGS__>           \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
    break;                \
  case 1024:                \
    kernel<1024,__VA_ARGS__>            \
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);    \
      break;                \
  default:                \
    errorCULQCD("%s not implemented for %d threads\n", #kernel, tp.block.x); \
    }





#define LAUNCH_KERNEL_GAUGEFIX_BORDER(kernel, tp, stream, array, points, relax_boost, parity, size, ...)     \
  if(tp.block.z==0){\
  switch (tp.block.x) {             \
  case 256:                \
    kernel<0, 32,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 512:                \
    kernel<0, 64,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 768:                \
    kernel<0, 96,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 1024:               \
    kernel<0, 128,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  default:                \
    errorCULQCD("%s not implemented for %d threads\n", #kernel, tp.block.x); \
    }\
  }\
  else if(tp.block.z==1){\
  switch (tp.block.x) {             \
  case 256:                \
    kernel<1, 32,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 512:                \
    kernel<1, 64,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 768:                \
    kernel<1, 96,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 1024:               \
    kernel<1, 128,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  default:                \
    errorCULQCD("%s not implemented for %d threads\n", #kernel, tp.block.x); \
    }\
  }\
  else if(tp.block.z==2){\
  switch (tp.block.x) {             \
  case 256:                \
    kernel<2, 32,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 512:                \
    kernel<2, 64,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 768:                \
    kernel<2, 96,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 1024:               \
    kernel<2, 128,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  default:                \
    errorCULQCD("%s not implemented for %d threads\n", #kernel, tp.block.x); \
    }\
  }\
   else if(tp.block.z==3){\
  switch (tp.block.x) {             \
  case 128:                \
    kernel<3, 32,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 256:                \
    kernel<3, 64,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 384:                \
    kernel<3, 96,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 512:               \
    kernel<3, 128,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;              \
  case 640:               \
    kernel<3, 160,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 768:               \
    kernel<3, 192,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;             \
  case 896:               \
    kernel<3, 224,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;               \
  case 1024:               \
    kernel<3, 256,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;     \
  default:                \
    errorCULQCD("%s not implemented for %d threads\n", #kernel, tp.block.x); \
    }\
}\
  else if(tp.block.z==4) {\
  switch (tp.block.x) {             \
  case 128:                \
    kernel<4, 32,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 256:                \
    kernel<4, 64,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 384:                \
    kernel<4, 96,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 512:               \
    kernel<4, 128,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;              \
  case 640:               \
    kernel<4, 160,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 768:               \
    kernel<4, 192,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;             \
  case 896:               \
    kernel<4, 224,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;               \
  case 1024:               \
    kernel<4, 256,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;     \
  default:                \
    errorCULQCD("%s not implemented for %d threads\n", #kernel, tp.block.x); \
    }\
}\
  else if(tp.block.z==5) {\
  switch (tp.block.x) {             \
  case 128:                \
    kernel<5, 32,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 256:                \
    kernel<5, 64,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 384:                \
    kernel<5, 96,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 512:               \
    kernel<5, 128,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;              \
  case 640:               \
    kernel<5, 160,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;                \
  case 768:               \
    kernel<5, 192,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;             \
  case 896:               \
    kernel<5, 224,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;               \
  case 1024:               \
    kernel<5, 256,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, points, relax_boost, parity, size);   \
    break;     \
  default:                \
    errorCULQCD("%s not implemented for %d threads\n", #kernel, tp.block.x); \
    }\
}\
else{\
    errorCULQCD("%s not implemented for type %d threads\n", #kernel, tp.block.z); \
}









#define LAUNCH_KERNEL_GAUGEFIX(kernel, tp, stream, array, relax_boost, parity, size, ...)     \
  if(tp.block.z==0){\
  switch (tp.block.x) {             \
  case 256:                \
    kernel<0, 32,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 512:                \
    kernel<0, 64,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 768:                \
    kernel<0, 96,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 1024:               \
    kernel<0, 128,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  default:                \
    errorCULQCD("%s not implemented for %d threads\n", #kernel, tp.block.x); \
    }\
  }\
  else if(tp.block.z==1){\
  switch (tp.block.x) {             \
  case 256:                \
    kernel<1, 32,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 512:                \
    kernel<1, 64,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 768:                \
    kernel<1, 96,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 1024:               \
    kernel<1, 128,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  default:                \
    errorCULQCD("%s not implemented for %d threads\n", #kernel, tp.block.x); \
    }\
  }\
  else if(tp.block.z==2){\
  switch (tp.block.x) {             \
  case 256:                \
    kernel<2, 32,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 512:                \
    kernel<2, 64,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 768:                \
    kernel<2, 96,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 1024:               \
    kernel<2, 128,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  default:                \
    errorCULQCD("%s not implemented for %d threads\n", #kernel, tp.block.x); \
    }\
  }\
   else if(tp.block.z==3){\
  switch (tp.block.x) {             \
  case 128:                \
    kernel<3, 32,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 256:                \
    kernel<3, 64,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 384:                \
    kernel<3, 96,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 512:               \
    kernel<3, 128,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;              \
  case 640:               \
    kernel<3, 160,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 768:               \
    kernel<3, 192,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;             \
  case 896:               \
    kernel<3, 224,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;               \
  case 1024:               \
    kernel<3, 256,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;     \
  default:                \
    errorCULQCD("%s not implemented for %d threads\n", #kernel, tp.block.x); \
    }\
}\
  else if(tp.block.z==4) {\
  switch (tp.block.x) {             \
  case 128:                \
    kernel<4, 32,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 256:                \
    kernel<4, 64,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 384:                \
    kernel<4, 96,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 512:               \
    kernel<4, 128,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;              \
  case 640:               \
    kernel<4, 160,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 768:               \
    kernel<4, 192,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;             \
  case 896:               \
    kernel<4, 224,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;               \
  case 1024:               \
    kernel<4, 256,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;     \
  default:                \
    errorCULQCD("%s not implemented for %d threads\n", #kernel, tp.block.x); \
    }\
}\
  else if(tp.block.z==5) {\
  switch (tp.block.x) {             \
  case 128:                \
    kernel<5, 32,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 256:                \
    kernel<5, 64,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 384:                \
    kernel<5, 96,__VA_ARGS__>           \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 512:               \
    kernel<5, 128,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;              \
  case 640:               \
    kernel<5, 160,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;                \
  case 768:               \
    kernel<5, 192,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;             \
  case 896:               \
    kernel<5, 224,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;               \
  case 1024:               \
    kernel<5, 256,__VA_ARGS__>            \
      <<< tp.grid, tp.block.x, tp.shared_bytes, stream >>>(array, relax_boost, parity, size);   \
    break;     \
  default:                \
    errorCULQCD("%s not implemented for %d threads\n", #kernel, tp.block.x); \
    }\
}\
else{\
    errorCULQCD("%s not implemented for type %d threads\n", #kernel, tp.block.z); \
}

