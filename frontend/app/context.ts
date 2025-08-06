import { createContext } from "react";
import { LocalApi } from "./client/local";

export const LocalAPIContext = createContext<LocalApi | undefined>(undefined);
