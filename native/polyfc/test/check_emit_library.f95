!CHECK case: exports
!CHECK offload-only
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -fstdpar-emit-library={output}.polyast -c -o {output}.o {input}
!CHECK do: polyfc --polyc {output}.polyast --list-exports
!CHECK requires@0: exportedA
!CHECK requires@1: exportedB

module emit_library_m
contains
    integer function helper(x)
        integer, intent(in) :: x
        helper = x * 2
    end function

    integer function unrelated(x)
        integer, intent(in) :: x
        unrelated = x + 99
    end function

    integer function exportedA(x) bind(c, name = "exportedA")
        integer, intent(in) :: x
        exportedA = helper(x) + 1
    end function

    integer function exportedB(x) bind(c, name = "exportedB")
        integer, intent(in) :: x
        exportedB = helper(x) + 2
    end function
end module emit_library_m
