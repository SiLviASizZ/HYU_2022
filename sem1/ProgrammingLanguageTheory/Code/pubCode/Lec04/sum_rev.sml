fun sum xs = 
    case xs of
         [] => 0
       | x::xs' => x + sum xs'

fun sum xs = 
let fun aux(xs, acc) =
in
    aux(  ,  )
end

(* [1,2,3] => [3, 2, 1] *)
fun rev xs =
    case xs of
         [] => []
       | x::xs' => (rev xs') @ [x] 

(* input: xs=[1, 2, 3] acc=[]
*
        1::[2,3]         1::[]
            2::[3]       2::[1]
                3::[]    3::[2, 1]
*)
fun rev xs =
let fun aux(xs, acc) =
in
    aux(  ,  )
end
