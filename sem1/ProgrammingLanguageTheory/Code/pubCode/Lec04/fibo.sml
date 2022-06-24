
fun fibo (0) = 1
  | fibo (1) = 1
  | fibo (n) = fibo(n-1) + fibo(n-2)

fun fibo_series(0) = [fibo(0)]
  | fibo_series(n) = fibo_series(n-1) @ [fibo(n)]

fun fibo_series2(0) = [fibo(0)]
  | fibo_series2(n) = fibo(n)::fibo_series2(n-1)

(* fibo3 returns a tuple (f_n, f_(n-1))
   e.g. fibo3 0 ==> (1, 1)
        fibo3 1 ==> (2, 1)
        fibo3 2 ==> (3, 2)
 *)
fun fibo3 (0) = (1, 1)
  | fibo3 (n) =
  let 
      val (p, pp) = fibo3(n-1)
  in
     (p+pp, p)
  end

(*
(* fibo_aux *)
(*
fun fibo2 (n) = 
let fun fibo_aux(0, ...) =
      | fibo_aux(c, ...) = 
in
    fibo_aux(...)
end
*)

     

*)
