datatype expr = NUM of int
            | PLUS of expr * expr
            | MINUS of expr * expr;

datatype formula = TRUE
            | FALSE
            | NOT of formula
            | ANDALSO of formula * formula
            | ORELSE of formula * formula
            | IMPLY of formula * formula
            | LESS of expr * expr;
                (* LESS(a, b) is true if a < b *)

(* eval function , takes a formula value and returns Boolean value of formula , formula -> bool *)
(* inplies is  *)

(* Problem 1 *)
fun eval x =
    case x of
        (x1, x2) => if xl then true else if x2 then true else false


(* Problem 2 *)
type name = string
datatype metro = STATION of name
                | AREA of name * metro
                | CONNECT of metro * metro



(* Problem 3 *)