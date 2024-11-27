#include <iostream>
#include <string>
#include <getopt.h>


namespace base {

struct SunwTerminalConfig {
  int Nx;
  int Ny;
  int Nz;
  int Nt;
  
  int traj;
  double beta;
  
  std::string input_file = "";
  std::string output_file = "";

  void detail () {
    std::cout << "Nx: " << Nx << std::endl;
    std::cout << "Ny: " << Ny << std::endl;
    std::cout << "Nz: " << Nz << std::endl;
    std::cout << "Nt: " << Nt << std::endl;
    std::cout << "traj: " << traj << std::endl;
    std::cout << "beta: " << beta << std::endl;
    std::cout << "input_file: " << input_file << std::endl;
    std::cout << "output_file: " << output_file << std::endl;
  }

  static SunwTerminalConfig parse_config(int argc, char *argv[]) {
    SunwTerminalConfig config;
    
    // long options
    struct option long_options[] = {
      {"Nx", required_argument, 0, 'x'},
      {"Ny", required_argument, 0, 'y'},
      {"Nz", required_argument, 0, 'z'},
      {"Nt", required_argument, 0, 't'},
      {"traj", required_argument, 0, 'T'},
      {"beta", required_argument, 0, 'b'},
      {"input", required_argument, 0, 'i'},
      {"output", required_argument, 0, 'o'},
      {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "x:y:z:t:T:b:i:o:", long_options, &option_index)) != -1) {
      switch (opt) {
        case 'x':
          config.Nx = atoi(optarg);
          break;
        case 'y':
          config.Ny = atoi(optarg);
          break;
        case 'z':
          config.Nz = atoi(optarg);
          break;
        case 't':
          config.Nt = atoi(optarg);
          break;
        case 'T':
          config.traj = atoi(optarg);
          break;
        case 'b':
          config.beta = atof(optarg);
          break;
        case 'i':
          config.input_file = optarg;
          break;
        case 'o':
          config.output_file = optarg;
          break;
        default:
          std::cout << "Invalid option: " << opt << std::endl;
          exit(1);
      }
    }
    return config;
  }
};
}