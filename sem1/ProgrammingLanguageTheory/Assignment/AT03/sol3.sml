datatype pattern = Wildcard | Variable of string | UnitP
                    | ConstP of int | TupleP of pattern list
                    | ConstructorP of string * pattern;

datatype valu = Const of int | Unit | Tuple of valu list
                | Constructor of string * valu

fun check_pat (p:pattern) = (* pattern -> bool *)
    let
        val lista = []
    in
        case p of
            Variable(x) => lista @ [x]
            | ConstructorP(x, y) => 
            | TupleP(tuple) => foldl 
    end