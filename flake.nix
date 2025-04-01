{
  description = "Tech for Low Vision";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;
        pythonPackages = python.pkgs;
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            pythonPackages.trimesh
            pythonPackages.numpy
            pythonPackages.matplotlib
            pythonPackages.scipy
            pythonPackages.flask
          ];
          shellHook = ''
            echo "Python development environment loaded."
          '';
        };
      }
    );
}
