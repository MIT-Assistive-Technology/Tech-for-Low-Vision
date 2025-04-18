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
              check-merge-conflicts.enable = true;
              commitizen.enable = true;
              convco.enable = true;
              forbid-new-submodules.enable = true;
              gitlint.enable = true;
              markdownlint.enable = true;
              mdformat.enable = true;
              mdsh.enable = true;
              deadnix.enable = true;
              flake-checker.enable = true;
              nil.enable = true;
              statix.enable = true;
              autoflake.enable = true;
              check-builtin-literals.enable = true;
              check-docstring-first.enable = true;
              check-python.enable = true;
              flynt.enable = true;
              isort.enable = true;
              name-tests-test.enable = true;
              pyright.enable = true;
              python-debug-statements.enable = true;
              pyupgrade.enable = true;
              ruff.enable = true;
              ruff-format.enable = true;
              sort-requirements-txt.enable = true;
              ripsecrets.enable = true;
              trufflehog.enable = true;
              shellcheck.enable = true;
              shfmt.enable = true;
              typos.enable = true;
              check-added-large-files.enable = true;
              check-case-conflicts.enable = true;
              check-executables-have-shebangs.enable = true;
              check-shebang-scripts-are-executable.enable = true;
              check-symlinks.enable = true;
              check-vcs-permalinks.enable = true;
              detect-private-keys.enable = true;
              end-of-file-fixer.enable = true;
              mixed-line-endings.enable = true;
              trim-trailing-whitespace.enable = true;
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
            pythonPackages.rtree
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
