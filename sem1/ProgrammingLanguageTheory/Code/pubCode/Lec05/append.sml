


fun append(alist, blist) =
    case alist of
         [] => blist
       | a::[] => a::blist
       | a::alist2 => a::append(alist2, blist)
