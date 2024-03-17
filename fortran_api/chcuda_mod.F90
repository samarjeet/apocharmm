module chcuda_mod
  use, intrinsic :: iso_c_binding
  implicit none
  public

  interface

    subroutine initialize_chcuda_fortran() bind(C, name="initialization_chcuda")
    end subroutine initialize_chcuda_fortran

    subroutine calculate_energy_chcuda_fortran() bind(C,name="calculate_energy_chcuda") 
    end subroutine calculate_energy_chcuda_fortran

    subroutine dynamics_chcuda_fortran() bind(C, name="dynamics_chcuda")
    end subroutine dynamics_chcuda_fortran

    subroutine minimize_chcuda_fortran() bind(C, name="minimization_chcuda")
    end subroutine minimize_chcuda_fortran

  end interface
contains

  subroutine initialize_chcuda()
    implicit none
    call initialize_chcuda_fortran()
  end subroutine initialize_chcuda

  ! Add the parameters
  subroutine energy_chcuda()
    implicit none
    write(*,*) "Calculating energy "
    call calculate_energy_chcuda_fortran()
  end subroutine energy_chcuda

  ! Add the parameters
  subroutine dynamc_chcuda()
    implicit none
    write(*,*) "dynamc from fortran"
    call dynamics_chcuda_fortran()
  end subroutine dynamc_chcuda

  ! Add the parameters
  subroutine minimize_chcuda()
    implicit none 
    write(*,*) "minimize from fortran"
    call minimize_chcuda_fortran()
  end subroutine

end module chcuda_mod
