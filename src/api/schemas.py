from pydantic import Field, create_model

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = ["Time"] + V_COLUMNS + ["Amount"]

_fields = {
    "Time": (float, Field(..., ge=0, description="Seconds since first transaction")),
    "Amount": (float, Field(..., ge=0)),
}
_fields.update({v: (float, Field(...)) for v in V_COLUMNS})

Transaction = create_model("Transaction", **_fields)
