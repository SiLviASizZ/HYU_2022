(* private function/variable in our library *)
(* callbacks have int->unit type *)
val cbs : (int->unit) list ref   = ref []

(* internal function to handle events *)
(*  (... ; ... ; ...)  *)
fun onEvent i =
   let fun loop fs =
        case fs of 



    in loop (!cbs) end

(* public interface of our library *)
(* append f to cbs *)
fun registerKeyEventFunc f =


(* example client code. *)

val timesPressed = ref 0
val _ = registerKeyEventFunc (
            fn _ =>  timesPressed := 1+(!timesPressed) (* increment timesPressed 1 *) )

(* print a message if the pressed key = i *)
(* use Int.toString and ^ *)
fun printIfPressed i =
    registerKeyEventFunc (fn j => if j=i
                        then print ("\027[34;1myou pressed " ^ Int.toString i ^ "\027[0m\n")
                        else ())

printIfPressed 4;
printIfPressed 11;
printIfPressed 23;
printIfPressed 4;

