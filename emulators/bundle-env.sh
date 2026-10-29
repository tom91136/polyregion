emu="${1//\\//}"
if [ -f "$emu/env.sh" ]; then
  source "$emu/env.sh"
else
  # windows bundle ships env.bat (cmd) only; reconstruct the same vars in bash
  export PATH="$(cygpath -u "$emu" 2>/dev/null || echo "$emu")/bin:$PATH"
  win() { cygpath -w "$1" 2>/dev/null || echo "$1"; }
  vk="$emu/share/vulkan/icd.d"; icd=$(win "$(echo "$vk"/lvp_icd.*.json)")
  [ -f "$vk/vk_swiftshader_icd.json" ] && icd="$icd;$(win "$vk/vk_swiftshader_icd.json")" || true
  export VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd" POLYINVOKE_OPENCL_CPU=1
  # XXX Windows searches PATH last so the bundled pocl and Vulkan loader lose to System32; load them by absolute path
  export POLYINVOKE_OPENCL_LIB="$(win "$emu/bin/OpenCL.dll")" POLYINVOKE_VULKAN_LIB="$(win "$emu/bin/vulkan-1.dll")"
fi
