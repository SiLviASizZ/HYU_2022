datatype pattern = Wildcard | Variable of string | UnitP
                    | ConstP of int | TupleP of pattern list
                    | ConstructorP of string * pattern;

datatype valu = Const of int | Unit | Tuple of valu list
                | Constructor of string * valu
