program main
  use chcuda_mod
  implicit none
  call energy_chcuda()

  call dynamc_chcuda()

  call minimize_chcuda()

  write(*,*) "Hello World from main"
end


