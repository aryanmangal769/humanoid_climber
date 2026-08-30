#include "EverestTerrainActor.h"
#include "ProceduralMeshComponent.h"
#include "KismetProceduralMeshLibrary.h"
#include "Dom/JsonObject.h"
#include "Materials/MaterialInterface.h"

AEverestTerrainActor::AEverestTerrainActor()
{
    PrimaryActorTick.bCanEverTick = false;
    Mesh = CreateDefaultSubobject<UProceduralMeshComponent>(TEXT("TerrainMesh"));
    SetRootComponent(Mesh);
    Mesh->SetCollisionEnabled(ECollisionEnabled::NoCollision); // MuJoCo owns robot contact.
    Mesh->bUseAsyncCooking = true;
}

void AEverestTerrainActor::ApplyTerrain(const TSharedPtr<FJsonObject>& T)
{
    if (!T.IsValid()) return;
    const int32 W = T->GetIntegerField(TEXT("grid_width"));
    const int32 H = T->GetIntegerField(TEXT("grid_height"));
    const double WidthM = T->GetNumberField(TEXT("world_width_m"));
    const double DepthM = T->GetNumberField(TEXT("world_depth_m"));
    const auto& Center = T->GetArrayField(TEXT("terrain_center"));
    const auto& Heights = T->GetArrayField(TEXT("heights"));
    if (W < 2 || H < 2 || Heights.Num() != W * H || Center.Num() < 3) return;

    const double Cx = Center[0]->AsNumber(), Cy = Center[1]->AsNumber();
    TArray<FVector> Vertices; Vertices.Reserve(W * H);
    TArray<FVector2D> UVs; UVs.Reserve(W * H);
    TArray<FLinearColor> Colors; Colors.Reserve(W * H);
    TArray<int32> Triangles; Triangles.Reserve((W - 1) * (H - 1) * 6);

    double MinZ = DBL_MAX, MaxZ = -DBL_MAX;
    for (const auto& V : Heights) { const double Z = V->AsNumber(); MinZ = FMath::Min(MinZ, Z); MaxZ = FMath::Max(MaxZ, Z); }
    const double RangeZ = FMath::Max(0.001, MaxZ - MinZ);

    for (int32 Y = 0; Y < H; ++Y)
    {
        const double MuY = Cy + (0.5 - double(Y) / double(H - 1)) * DepthM;
        for (int32 X = 0; X < W; ++X)
        {
            const int32 I = Y * W + X;
            const double MuX = Cx + (double(X) / double(W - 1) - 0.5) * WidthM;
            const double Z = Heights[I]->AsNumber();
            Vertices.Emplace(MuX * 100.0, -MuY * 100.0, Z * 100.0);
            UVs.Emplace(double(X) / double(W - 1), double(Y) / double(H - 1));
            const float Elev = float((Z - MinZ) / RangeZ);
            const FLinearColor Rock(0.10f, 0.115f, 0.13f, 1.0f);
            const FLinearColor Ice(0.52f, 0.64f, 0.72f, 1.0f);
            const FLinearColor Snow(0.88f, 0.93f, 0.97f, 1.0f);
            Colors.Add(Elev > 0.58f ? FMath::Lerp(Ice, Snow, (Elev - 0.58f) / 0.42f) : FMath::Lerp(Rock, Ice, Elev / 0.58f));
        }
    }
    for (int32 Y = 0; Y < H - 1; ++Y) for (int32 X = 0; X < W - 1; ++X)
    {
        const int32 A=Y*W+X, B=A+1, C=A+W, D=C+1;
        Triangles.Append({A,C,B, B,C,D});
    }
    TArray<FVector> Normals; TArray<FProcMeshTangent> Tangents;
    UKismetProceduralMeshLibrary::CalculateTangentsForMesh(Vertices, Triangles, UVs, Normals, Tangents);
    Mesh->ClearAllMeshSections();
    Mesh->CreateMeshSection_LinearColor(0, Vertices, Triangles, Normals, UVs, Colors, Tangents, false);
    if (UMaterialInterface* Mat = LoadObject<UMaterialInterface>(nullptr, TEXT("/Game/Everest/Materials/M_EverestTerrain.M_EverestTerrain"))) Mesh->SetMaterial(0, Mat);
}
