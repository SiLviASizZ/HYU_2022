fun merge (xs, ys) =
case (xs, ys) of
    ([], ys) => ys
    | (xs, []) => xs
    | (x::xs', y::ys') => if x < y then x::merge(xs', ys) else y::merge(xs, ys');

fun reverse xs =
    let
        fun reverse' (zs, ys) =
            case zs of
                [] => ys
                | z::zs' => reverse'(zs', z::ys)
    in
        reverse' (xs, [])
    end

fun pi (a, b, f) =
