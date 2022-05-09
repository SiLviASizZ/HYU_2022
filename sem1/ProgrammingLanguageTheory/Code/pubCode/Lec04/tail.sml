
(* examples of making functions tail-recursive *)

fun sum1 xs =
    case xs of
        [] => 0
      | i::xs' => i + sum1(xs')

(* tail recursive version *)
fun sum2 xs =
    let fun aux(xs, acc) =
    in
    end


fun rev1 xs =
   case xs of
       [] => []
     | x::xs' => (rev1 xs') @ [x]


(* input: xs=[1, 2, 3] acc=[]
*
       xs= 1::[2,3]    acc= 1::[]
           2::[3]           2::[1]
           3::[]            3::[2, 1]
*)

(* tail recursive version *)
fun rev2 xs =
    let fun aux(xs,acc) =
    in
        
    end
