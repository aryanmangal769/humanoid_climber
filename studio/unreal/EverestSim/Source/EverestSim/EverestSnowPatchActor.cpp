#include "EverestSnowPatchActor.h"
#include "ProceduralMeshComponent.h"
#include "KismetProceduralMeshLibrary.h"
#include "Dom/JsonObject.h"
#include "Materials/MaterialInterface.h"

AEverestSnowPatchActor::AEverestSnowPatchActor()
{
    PrimaryActorTick.bCanEverTick=false;
    Mesh=CreateDefaultSubobject<UProceduralMeshComponent>(TEXT("NewtonSnowSurface"));
    SetRootComponent(Mesh); Mesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
}

void AEverestSnowPatchActor::ApplySnowFrame(const TSharedPtr<FJsonObject>& F)
{
    if (!F.IsValid() || F->GetStringField(TEXT("mode")) != TEXT("live")) { Mesh->SetVisibility(false); return; }
    const auto& R=F->GetArrayField(TEXT("resolution")); const auto& O=F->GetArrayField(TEXT("origin")); const auto& S=F->GetArrayField(TEXT("size"));
    const auto& Hs=F->GetArrayField(TEXT("heights")); const auto& Cs=F->GetArrayField(TEXT("compaction"));
    if (R.Num()<2 || O.Num()<3 || S.Num()<2) return;
    const int32 W=(int32)R[0]->AsNumber(), H=(int32)R[1]->AsNumber(); if (W<2||H<2||Hs.Num()!=W*H) return;
    TArray<FVector> V; TArray<FVector2D> UV; TArray<FLinearColor> Col; TArray<int32> T; V.Reserve(W*H); UV.Reserve(W*H); Col.Reserve(W*H);
    const double Ox=O[0]->AsNumber(), Oy=O[1]->AsNumber(), Wm=S[0]->AsNumber(), Hm=S[1]->AsNumber();
    for(int32 Y=0;Y<H;++Y) for(int32 X=0;X<W;++X){ const int32 I=Y*W+X; const double Xw=Ox+double(X)/(W-1)*Wm; const double Yw=Oy+double(Y)/(H-1)*Hm; const double Zw=Hs[I]->AsNumber()+0.003; V.Emplace(Xw*100,-Yw*100,Zw*100); UV.Emplace(double(X)/(W-1),double(Y)/(H-1)); const float C=Cs.IsValidIndex(I)?FMath::Clamp((float)Cs[I]->AsNumber(),0.f,1.f):0.f; Col.Emplace(0.96f-0.18f*C,0.98f-0.12f*C,1.0f-0.05f*C,1); }
    for(int32 Y=0;Y<H-1;++Y) for(int32 X=0;X<W-1;++X){int32 A=Y*W+X,B=A+1,C=A+W,D=C+1;T.Append({A,C,B,B,C,D});}
    TArray<FVector>N; TArray<FProcMeshTangent>Tan; UKismetProceduralMeshLibrary::CalculateTangentsForMesh(V,T,UV,N,Tan);
    Mesh->ClearAllMeshSections(); Mesh->CreateMeshSection_LinearColor(0,V,T,N,UV,Col,Tan,false); Mesh->SetVisibility(true);
    if(UMaterialInterface* M=LoadObject<UMaterialInterface>(nullptr,TEXT("/Game/Everest/Materials/M_NewtonSnow.M_NewtonSnow"))) Mesh->SetMaterial(0,M);
}
