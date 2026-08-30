#pragma once

#include "CoreMinimal.h"
#include "Dom/JsonObject.h"

namespace EverestProtocol
{
    inline FVector ToUnrealPosition(const TArray<TSharedPtr<FJsonValue>>& A)
    {
        if (A.Num() < 3) return FVector::ZeroVector;
        return FVector(A[0]->AsNumber() * 100.0, -A[1]->AsNumber() * 100.0, A[2]->AsNumber() * 100.0);
    }

    inline FQuat ToUnrealQuat(const TArray<TSharedPtr<FJsonValue>>& A)
    {
        // MuJoCo sends wxyz in RH coordinates. Reflect Y into Unreal's LH basis.
        if (A.Num() < 4) return FQuat::Identity;
        const double W=A[0]->AsNumber(), X=A[1]->AsNumber(), Y=A[2]->AsNumber(), Z=A[3]->AsNumber();
        return FQuat(-X, Y, -Z, W).GetNormalized();
    }
}
