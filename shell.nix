let
  # We pin to a specific nixpkgs commit for reproducibility.
  # Last updated: 2026-02-10. Check for new commits at https://status.nixos.org.
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/refs/tags/25.11.tar.gz") { };
in
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (
      python-pkgs: with python-pkgs; [
        # select Python packages here
        pettingzoo
        grpcio
        grpcio-tools
        ruff
      ]
    ))
  ];
}
