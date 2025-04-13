{
  description = "Tech for Low Vision";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    pre-commit-hooks.url = "github:cachix/git-hooks.nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      pre-commit-hooks,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;
        pythonPackages = python.pkgs;
      in
      {
        checks = {
          pre-commit-check = pre-commit-hooks.lib.${system}.run {
            src = ./.;
            hooks = {
              nixfmt-rfc-style.enable = true;
            };
          };
        };

        devShells.default = pkgs.mkShell {
          inherit (self.checks.${system}.pre-commit-check) shellHook;
          buildInputs = self.checks.${system}.pre-commit-check.enabledPackages ++ [
            python
            pythonPackages.trimesh
            pythonPackages.numpy
            pythonPackages.matplotlib
            pythonPackages.scipy
            pythonPackages.flask
          ];
        };

        apps = {
          default =
            let
              runner = pkgs.writeScriptBin "run" ''
                #!${pkgs.bash}/bin/bash
                ${pythonPackages.flask}/bin/flask run
              '';
            in
            {
              type = "app";
              program = "${runner}/bin/run";
            };
        };

        formatter = nixpkgs.legacyPackages.${system}.nixfmt-rfc-style;
      }
    );
}
