import * as React from "react";

import { cn } from "@/lib/utils";

type SwitchProps = {
  checked: boolean;
  disabled?: boolean;
  label?: string;
  onCheckedChange: (checked: boolean) => void;
};

export function Switch({ checked, disabled, label, onCheckedChange }: SwitchProps) {
  return (
    <button
      type="button"
      disabled={disabled}
      aria-pressed={checked}
      onClick={() => onCheckedChange(!checked)}
      className={cn(
        "inline-flex cursor-pointer items-center gap-2 text-sm font-semibold text-stone-600 transition-all hover:text-stone-950 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50",
      )}
    >
      <span
        className={cn(
          "inline-flex h-6 w-11 items-center rounded-full border p-0.5 shadow-inner transition",
          checked ? "border-emerald-600/30 bg-emerald-100" : "border-stone-200 bg-stone-100",
        )}
      >
        <span className={cn("h-4.5 w-4.5 rounded-full transition", checked ? "translate-x-5 bg-emerald-700" : "bg-stone-500")} />
      </span>
      {label ? <span>{label}</span> : null}
    </button>
  );
}
