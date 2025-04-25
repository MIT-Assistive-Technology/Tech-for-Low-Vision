{
  description = "Tech for Low Vision";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    mkElmDerivation.url = "github:jeslie0/mkElmDerivation";
    pre-commit-hooks.url = "github:cachix/git-hooks.nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      mkElmDerivation,
      pre-commit-hooks,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          overlays = [ mkElmDerivation.overlays.mkElmDerivation ];
          inherit system;
        };
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
              elm-format.enable = true;
              check-yaml.enable = true;
              yamlfmt.enable = true;
              yamllint.enable = true;
              check-json.enable = true;
              pretty-format-json.enable = true;
              tagref.enable = true;
            };
          };
        };

        devShells.default = pkgs.mkShell {
          inherit (self.checks.${system}.pre-commit-check) shellHook;
          buildInputs = self.checks.${system}.pre-commit-check.enabledPackages ++ [
            # Python backend
            python
            pythonPackages.trimesh
            pythonPackages.numpy
            pythonPackages.matplotlib
            pythonPackages.scipy
            pythonPackages.flask
            pythonPackages.rtree
            pythonPackages.pillow
            pythonPackages.pyserial

            # Elm frontend
            pkgs.elmPackages.elm
            pkgs.elmPackages.elm-review
            pkgs.elmPackages.elm-json
            pkgs.static-server
            pkgs.entr
          ];
        };

        apps = {
          default =
            let
              runner = pkgs.writeScriptBin "run" ''
                #!${pkgs.bash}/bin/bash
                trap 'kill 0' SIGINT
                ${pkgs.bash}/bin/bash serve/watch.sh &
                ${pkgs.static-server}/bin/static-server -host localhost -port 3000 serve &
                ${pythonPackages.flask}/bin/flask run &
                wait
              '';
            in
            {
              type = "app";
              program = "${runner}/bin/run";
            };
        };

        packages = {
          default = pkgs.mkElmDerivation {
            name = "tech-for-low-vision";
            src = ./serve;
            elmJson = serve/elm.json;
            nativeBuildInputs = [ pkgs.elmPackages.elm ];
            buildPhase = ''
              elm make src/Main.elm --output Main.js --optimize
            '';
            installPhase = ''
              mkdir -p $out/dist/elm
              cp Main.js $out/dist/elm
              sed 's/..compiled\/Main.js/\/elm\/Main.js/' src/index.html > $out/dist/index.html
              cp -r assets $out/dist/assets
            '';
          };
        };

        formatter = nixpkgs.legacyPackages.${system}.nixfmt-rfc-style;
      }
    );
}
